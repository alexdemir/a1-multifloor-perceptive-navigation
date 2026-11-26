import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_dir = get_package_share_directory('a1_description')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # Default file paths (starts on Floor 1; FloorManager switches across N floors via building_topology.yaml)
    default_map = os.path.join(pkg_dir, 'maps', 'floor1_map.yaml')
    topology_file = os.path.join(pkg_dir, 'config', 'building_topology.yaml')
    default_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    default_rviz_config = os.path.join(pkg_dir, 'rviz', 'nav2.rviz')
    if not os.path.exists(default_rviz_config):
        default_rviz_config = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    # Launch Configurations
    sim = LaunchConfiguration('sim')
    gui = LaunchConfiguration('gui')
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_rviz = LaunchConfiguration('use_rviz')
    actors = LaunchConfiguration('actors')
    mapping = LaunchConfiguration('mapping')

    declare_sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='true',
        description='Set to "true" to launch Gazebo alongside Nav2, or "false" if Gazebo is already running'
    )
    declare_gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Set to "true" to run Gazebo with GUI, or "false" for headless'
    )
    declare_map_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map,
        description='Full path to initial map yaml file'
    )
    declare_params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Full path to nav2 parameters file'
    )
    declare_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )
    declare_autostart_arg = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup nav2 stack'
    )
    declare_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz with Nav2 navigation config'
    )
    declare_actors_arg = DeclareLaunchArgument(
        'actors',
        default_value='true',
        description='Set to "true" to move dynamic obstacle actors on Floor 1, 2, and 3'
    )
    declare_mapping_arg = DeclareLaunchArgument(
        'mapping',
        default_value='false',
        description='Set to "true" to run dlio_map_node & /save_pcd during autonomous mapping patrol'
    )

    # 1. Gazebo Simulation with Unitree A1 + DLIO Odometry (active when sim:=true)
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'gui': gui,
            'dlio': 'true',
            'mapping': mapping,
            'rviz': 'false'
        }.items(),
        condition=IfCondition(sim)
    )

    # 2. Static Transform Publisher: map -> odom (REP-105 standard)
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'map', '--child-frame-id', 'odom'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # 3. Robot-Centric 3D LiDAR Obstacle Filter (/ouster/points -> /ouster/points_filtered)
    lidar_filter_node = Node(
        package='a1_description',
        executable='lidar_obstacle_filter.py',
        name='lidar_obstacle_filter',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 4. Nav2 Map Server
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'yaml_filename': map_yaml_file
    }
    configured_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites=param_substitutions,
        convert_types=True
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )

    map_server_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'node_names': ['map_server']
        }]
    )

    # 5. Nav2 Navigation Stack
    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': params_file,
            'use_composition': 'False',
            'use_respawn': 'False'
        }.items()
    )

    # 6. Topological N-Floor Manager (Dijkstra cross-floor routing + dynamic map switching)
    floor_manager_node = Node(
        package='a1_description',
        executable='floor_manager_node.py',
        name='floor_manager_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'topology_file': topology_file,
        }],
        output='screen'
    )

    # 7. Dynamic Moving Obstacle Actors (Floor 1 & Floor 2)
    dynamic_obstacles_node = Node(
        package='a1_description',
        executable='dynamic_obstacles_node.py',
        name='dynamic_obstacles_node',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(actors),
        output='screen'
    )

    # 8. RViz2 with Nav2 Configuration
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='nav2_rviz2',
        arguments=['-d', default_rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
        output='screen'
    )

    # 9. Compact Multi-Floor Goal Selector GUI (RViz Publish Point -> /multi_floor_goal)
    floor_goal_gui = Node(
        package='a1_description',
        executable='floor_goal_gui.py',
        name='floor_goal_gui',
        parameters=[{
            'use_sim_time': use_sim_time,
            'topology_file': topology_file,
        }],
        condition=IfCondition(use_rviz),
        output='screen'
    )

    return LaunchDescription([
        declare_sim_arg,
        declare_gui_arg,
        declare_map_arg,
        declare_params_arg,
        declare_sim_time_arg,
        declare_autostart_arg,
        declare_rviz_arg,
        declare_actors_arg,
        declare_mapping_arg,
        gazebo_sim,
        static_map_to_odom,
        lidar_filter_node,
        map_server_node,
        map_server_lifecycle,
        nav2_navigation,
        floor_manager_node,
        dynamic_obstacles_node,
        rviz_node,
        floor_goal_gui
    ])
