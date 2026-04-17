# Launch file: Training only (no Gazebo, pure Python env).


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("policy", default_value="ensemble"),
        DeclareLaunchArgument("env", default_value="A"), # train on A test on B and others
        DeclareLaunchArgument("config", default_value="config/train_ensemble.yaml"),

        Node(
            package="uncertainty_nav",
            executable="python3",
            name="trainer",
            arguments=["scripts/train/ppo_trainer.py", LaunchConfiguration("config")],
            output="screen",
        ),
    ])
