# PPO trainer for PolicyNetwork, VanillaMLP, RecurrentPolicy, LargeMLPPolicy.

import os
if os.environ.get("PARALLEL_TRAIN"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if os.environ.get("PARALLEL_TRAIN"):
    torch.set_num_threads(1)

import yaml
import json
from torch.utils.tensorboard import SummaryWriter

from uncertainty_nav.models import PolicyNetwork, ValueNetwork
from uncertainty_nav.baselines import VanillaMLP, RecurrentPolicy, LargeMLPPolicy
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B


class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def clear(self):
        self.__init__()

    def add(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        returns, advantages = [], []
        gae = 0.0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return returns, advantages


class PPOTrainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed everything for reproducibility — different seeds = different members
        seed = self.cfg.get("seed", 0)
        torch.manual_seed(seed)
        np.random.seed(seed)

        env_cfg = ENV_A if self.cfg.get("env", "A") == "A" else ENV_B
        self.env = PartialObsNavEnv(env_cfg, seed=seed)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        policy_type = self.cfg["policy_type"]

        if policy_type == "ensemble_member":
            # Each ensemble member is just a PolicyNetwork trained independently
            self.policy = PolicyNetwork(
                obs_dim, act_dim, self.cfg.get("hidden", 256)
            ).to(self.device)
        elif policy_type in ("lstm", "gru"):
            self.policy = RecurrentPolicy(
                obs_dim, act_dim,
                hidden=self.cfg.get("hidden", 256),
                rnn_type=policy_type,
            ).to(self.device)
        elif policy_type == "large_mlp":
            self.policy = LargeMLPPolicy(
                obs_dim, act_dim,
                n_members=self.cfg.get("n_members", 5),
                hidden=self.cfg.get("hidden", 256),
            ).to(self.device)
        else:
            self.policy = VanillaMLP(
                obs_dim, act_dim, self.cfg.get("hidden", 256)
            ).to(self.device)

        self.value_net = ValueNetwork(obs_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=self.cfg.get("lr", 3e-4),
        )

        self.buffer = RolloutBuffer()
        self.writer = SummaryWriter(log_dir=self.cfg.get("log_dir", "runs/default"))
        self.is_recurrent = policy_type in ("lstm", "gru")
        self.results = []
        self._ep_successes = []
        self._ep_collisions = []
        self._ep_spls = []

    def collect_rollout(self, n_steps=2048):
        self.buffer.clear()
        obs, _ = self.env.reset()
        hidden = self.policy.init_hidden() if self.is_recurrent else None
        episode_rewards = []
        ep_reward = 0.0

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.is_recurrent:
                    action, log_prob, hidden = self.policy.sample(obs_t, hidden)
                else:
                    action, log_prob = self.policy.sample(obs_t)
                value = self.value_net(obs_t).item()

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(obs, action_np, reward, log_prob.item(), value, float(done))
            ep_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                self._ep_successes.append(float(info.get("success", False)))
                self._ep_collisions.append(float(info.get("collision", False)))
                self._ep_spls.append(self.env.compute_spl() if info.get("success") else 0.0)
                ep_reward = 0.0
                obs, _ = self.env.reset()
                if self.is_recurrent:
                    hidden = self.policy.init_hidden()

        return episode_rewards

    def update(self, returns, advantages, clip_eps=0.2, epochs=10, entropy_coef=0.01):
        obs_t = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
        act_t = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_lp = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        bptt_len = self.cfg.get("bptt_len", 16)  # BPTT window length

        for _ in range(epochs):
            if self.is_recurrent:
                dones_t = torch.FloatTensor(self.buffer.dones).to(self.device)
                hidden = self.policy.init_hidden(1)
                means, log_stds = [], []
                for i in range(obs_t.shape[0]):
                    mean_i, log_std_i, hidden = self.policy(obs_t[i:i+1], hidden)
                    means.append(mean_i)
                    log_stds.append(log_std_i)
                    if i < len(dones_t) and dones_t[i] > 0.5:
                        hidden = self.policy.init_hidden(1)
                    elif (i + 1) % bptt_len == 0:
                        hidden = tuple(h.detach() for h in hidden)
                mean = torch.cat(means, dim=0)
                log_std = torch.cat(log_stds, dim=0)
            else:
                mean, log_std = self.policy(obs_t)

            dist = torch.distributions.Normal(mean, log_std.exp())
            act_clamped = act_t.clamp(-0.9999, 0.9999)
            raw = torch.atanh(act_clamped)
            new_lp = (dist.log_prob(raw) - torch.log(1 - act_t.pow(2) + 1e-6)).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = self.value_net(obs_t).squeeze()
            value_loss = nn.functional.mse_loss(value_pred, ret_t)

            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        total_steps = self.cfg.get("total_steps", 1_000_000)
        n_steps = self.cfg.get("n_steps", 2048)
        update_count = 0

        while update_count * n_steps < total_steps:
            ep_rewards = self.collect_rollout(n_steps)

            obs_last = torch.FloatTensor(np.array(self.buffer.obs[-1:])).to(self.device)
            last_val = self.value_net(obs_last).item()
            returns, advantages = self.buffer.compute_returns(last_val)
            p_loss, v_loss = self.update(returns, advantages)

            mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            sr = float(np.mean(self._ep_successes[-50:])) if self._ep_successes else 0.0
            cr = float(np.mean(self._ep_collisions[-50:])) if self._ep_collisions else 0.0
            spl = float(np.mean(self._ep_spls[-50:])) if self._ep_spls else 0.0
            step = update_count * n_steps

            self.writer.add_scalar("train/mean_reward", mean_reward, step)
            self.writer.add_scalar("train/success_rate", sr, step)
            self.writer.add_scalar("train/collision_rate", cr, step)
            self.writer.add_scalar("train/spl", spl, step)
            self.writer.add_scalar("train/policy_loss", p_loss, step)
            self.writer.add_scalar("train/value_loss", v_loss, step)

            self.results.append({
                "step": step, "mean_reward": mean_reward,
                "success_rate": sr, "collision_rate": cr, "mean_spl": spl,
            })

            if update_count % 10 == 0:
                print(f"[{step:>8}] reward={mean_reward:6.2f} | "
                      f"SR={sr:.2f} | coll={cr:.2f} | SPL={spl:.2f} | "
                      f"p_loss={p_loss:.4f}")

            update_count += 1

        self._save()

    def _save(self):
        out_dir = self.cfg.get("output_dir", "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        name = self.cfg.get("checkpoint_name", self.cfg["policy_type"])
        torch.save(self.policy.state_dict(), f"{out_dir}/{name}_policy.pt")
        with open(f"{out_dir}/{name}_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved to {out_dir}/{name}_policy.pt")


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "src/uncertainty_nav/config/train_ensemble.yaml"
    PPOTrainer(config).train()
