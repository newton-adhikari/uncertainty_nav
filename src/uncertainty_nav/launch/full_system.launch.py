from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
)

from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    Command,
    FindExecutable,
    LaunchConfiguration
)

from launch.conditions import IfCondition, UnlessCondition
from launch.actions import OpaqueFunction
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_uncertainty_nav = FindPackageShare("uncertainty_nav")
    pkg_gazebo_ros      = FindPackageShare("gazebo_ros")

    env_arg = DeclareLaunchArgument(
        "env", default_value="A",
        description="Environment: A (training, moderate noise) or B (test, high noise)"
    )
    policy_arg = DeclareLaunchArgument(
        "policy", default_value="ensemble",
        description="Policy type: ensemble | vanilla | lstm | gru | large_mlp"
    )
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint", default_value="",
        description="Absolute path to policy .pt checkpoint file"
    )
    unc_threshold_arg = DeclareLaunchArgument(
        "uncertainty_threshold", default_value="0.3",
        description="Epistemic uncertainty threshold for cautious behavior"
    )
    caution_scale_arg = DeclareLaunchArgument(
        "caution_scale", default_value="0.5",
        description="Action scale factor when uncertainty > threshold"
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true",
        description="Launch RViz2"
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true"
    )

    use_rviz       = LaunchConfiguration("use_rviz")
    use_sim_time   = LaunchConfiguration("use_sim_time")

    world_file_a = PathJoinSubstitution([pkg_uncertainty_nav, "worlds", "env_a.world"])
    world_file_b = PathJoinSubstitution([pkg_uncertainty_nav, "worlds", "env_b.world"])


    # Env A: laser_noise_std=0.01 (real LDS-01 accuracy)
    # Env B: laser_noise_std=0.05 (5× harder)
    urdf_xacro = PathJoinSubstitution([
        pkg_uncertainty_nav, "urdf", "turtlebot3_waffle_pi_uncertainty.urdf.xacro"
    ])

    robot_description_a = Command([
        FindExecutable(name="xacro"), " ", urdf_xacro,
        " laser_noise_std:=0.01"
    ])
    robot_description_b = Command([
        FindExecutable(name="xacro"), " ", urdf_xacro,
        " laser_noise_std:=0.05"
    ])

    def select_gazebo(context, *args, **kwargs):
        env_val = LaunchConfiguration("env").perform(context)
        world   = world_file_a if env_val != "B" else world_file_b
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([pkg_gazebo_ros, "launch", "gazebo.launch.py"])
                ]),
                launch_arguments={"world": world, "verbose": "false"}.items(),
            )
        ]

    def select_robot_description(context, *args, **kwargs):
        env_val = LaunchConfiguration("env").perform(context)
        desc    = robot_description_a if env_val != "B" else robot_description_b
        return [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{
                    "robot_description": ParameterValue(desc, value_type=str),
                    "use_sim_time": use_sim_time,
                }],
            )
        ]

    def select_agent_params(context, *args, **kwargs):
        env_val = LaunchConfiguration("env").perform(context)
        # Env A: full FoV, no extra occlusion
        # Env B: 90° FoV, 20% occlusion, 5% dropout
        if env_val == "B":
            fov_deg      = 90.0
            occ_prob     = 0.2
            drop_prob    = 0.05
        else:
            fov_deg      = 360.0
            occ_prob     = 0.0
            drop_prob    = 0.0
        return [
            Node(
                package="uncertainty_nav",
                executable="uncertainty_agent",
                name="uncertainty_agent",
                output="screen",
                parameters=[{
                    "checkpoint":            LaunchConfiguration("checkpoint"),
                    "uncertainty_threshold": LaunchConfiguration("uncertainty_threshold"),
                    "caution_scale":         LaunchConfiguration("caution_scale"),
                    "n_laser_beams":         36,
                    "fov_deg":               fov_deg,
                    "occlusion_prob":        occ_prob,
                    "dropout_prob":          drop_prob,
                    "map_size":              10.0,
                    "use_sim_time":          use_sim_time,
                }],
            )
        ]

    # Spawn TurtleBot3 in Gazebo 
    def spawn_turtlebot(context, *args, **kwargs):
        env_val = LaunchConfiguration("env").perform(context)
        desc    = robot_description_a if env_val != "B" else robot_description_b
        return [
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                name="spawn_turtlebot3",
                output="screen",
                arguments=[
                    "-entity",    "turtlebot3_waffle_pi",
                    "-topic",     "robot_description",
                    "-x",         "0.0",
                    "-y",         "0.0",
                    "-z",         "0.01",
                    "-Y",         "0.0",
                ],
            )
        ]

    # Particle filter node 
    particle_filter_node = Node(
        package="uncertainty_nav",
        executable="particle_filter",
        name="particle_filter_node",
        output="screen",
        parameters=[{
            "n_particles":       500,
            "map_size":          10.0,
            "n_scan_beams_used": 36,
            "obs_noise_std":     0.2,
            "use_sim_time":      use_sim_time,
        }],
    )

    # RViz uncertainty visualization node 
    rviz_uncertainty_node = Node(
        package="uncertainty_nav",
        executable="rviz_uncertainty",
        name="rviz_uncertainty_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # RViz2 
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d", PathJoinSubstitution([
                pkg_uncertainty_nav, "config", "uncertainty_nav.rviz"
            ]),
            "--ros-args", "--log-level", "rviz2:=warn",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    # Joint state publisher (for URDF visualization) 
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        # Arguments
        env_arg, policy_arg, checkpoint_arg,
        unc_threshold_arg, caution_scale_arg,
        use_rviz_arg, use_sim_time_arg,

        # Gazebo (world selected by env arg)
        OpaqueFunction(function=select_gazebo),

        # Robot description (noise level selected by env arg)
        OpaqueFunction(function=select_robot_description),
        joint_state_publisher,

        # Spawn TurtleBot3
        OpaqueFunction(function=spawn_turtlebot),

        # Agent (partial obs params selected by env arg)
        OpaqueFunction(function=select_agent_params),

        # Other nodes
        particle_filter_node,
        rviz_uncertainty_node,
        rviz2_node,
    ])