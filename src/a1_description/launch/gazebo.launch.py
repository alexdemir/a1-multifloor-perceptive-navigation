import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    dlio_prefix = '/home/alex/a1_navigation_ws/install/direct_lidar_inertial_odometry'
    if os.path.exists(dlio_prefix):
        for env_var, subdir in [
            ('AMENT_PREFIX_PATH', ''),
            ('CMAKE_PREFIX_PATH', ''),
            ('LD_LIBRARY_PATH', 'lib'),
        ]:
            p = os.path.join(dlio_prefix, subdir) if subdir else dlio_prefix
            cur = os.environ.get(env_var, '')
            if p not in cur.split(':'):
                os.environ[env_var] = p + (':' + cur if cur else '')

    pkg_dir = get_package_share_directory('a1_description')
    dlio_pkg_dir = get_package_share_directory('direct_lidar_inertial_odometry')
    install_dir = os.path.dirname(pkg_dir)
    
    # Configure Gazebo model paths for meshes
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += ':' + install_dir
    else:
        os.environ['GAZEBO_MODEL_PATH'] = install_dir

    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Set to "false" to run Gazebo headless without GUI'
    )
    dlio_arg = DeclareLaunchArgument(
        'dlio',
        default_value='true',
        description='Set to "false" to disable DLIO during rosbag recording for max FPS'
    )
    mapping_arg = DeclareLaunchArgument(
        'mapping',
        default_value='true',
        description='Set to "true" to run dlio_map_node (3D global map builder & /save_pcd service)'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Set to "false" to disable RViz during rosbag recording for max FPS'
    )
    gui = LaunchConfiguration('gui')
    dlio = LaunchConfiguration('dlio')
    mapping = LaunchConfiguration('mapping')
    rviz_cfg = LaunchConfiguration('rviz')

    world_file = os.path.join(pkg_dir, 'worlds', 'multi_floor_building.world')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'a1_with_lidar.xacro')
    rviz_file = os.path.join(pkg_dir, 'rviz', 'a1_lidar.rviz')
    dlio_yaml_path = os.path.join(dlio_pkg_dir, 'cfg', 'dlio.yaml')
    dlio_params_yaml_path = os.path.join(dlio_pkg_dir, 'cfg', 'params.yaml')

    # Process Xacro into URDF XML
    doc = xacro.process_file(xacro_file)
    robot_description = {'robot_description': doc.toxml()}

    # 1. Gazebo Server and GUI
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_file, 'gui': gui}.items()
    )

    # 2. Robot State Publisher (TF Tree: base -> trunk -> lidar_link, imu_link, legs)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # 3. Delayed Gazebo Robot Spawn (z=0.42 clears full straight-leg length; soft stand-up in RL node prevents any jerk/flip)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'a1_quadruped', '-x', '0.0', '-y', '0.0', '-z', '0.42'],
        output='screen'
    )
    delayed_spawn_entity = TimerAction(
        period=3.0,
        actions=[spawn_entity]
    )

    # 4. ros2_control Controller Spawners (wait automatically for controller_manager and activate immediately on spawn)
    spawn_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    spawn_jpc = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_group_position_controller'],
        output='screen'
    )

    # 5. RL Locomotion Controller (Isaac Gym PPO Inference Node)
    rl_controller = Node(
        package='a1_description',
        executable='rl_locomotion_controller.py',
        name='rl_locomotion_controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # 6. DLIO Odometry Node (Active when dlio:=true)
    dlio_odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': True}],
        remappings=[
            ('pointcloud', '/ouster/points'),
            ('imu', '/imu/data'),
            ('odom', 'dlio/odom_node/odom'),
            ('pose', 'dlio/odom_node/pose'),
            ('path', 'dlio/odom_node/path'),
            ('kf_pose', 'dlio/odom_node/keyframes'),
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
        ],
        condition=IfCondition(dlio)
    )

    # 6b. Robot-Centric 2.5D Elevation Map Node (fuses /ouster/points + DLIO odom -> /elevation_map/height_scan)
    elevation_map_node = Node(
        package='a1_description',
        executable='elevation_map_node.py',
        name='elevation_map_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(dlio)
    )

    # 7. DLIO Mapping Node (Active when mapping:=true)
    dlio_map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': True}],
        remappings=[
            ('keyframes', 'dlio/odom_node/pointcloud/keyframe'),
            ('map', 'dlio/map_node/map'),
        ],
        condition=IfCondition(mapping)
    )

    # 8. RViz2 (Launched when rviz:=true)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file] if os.path.exists(rviz_file) else [],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz_cfg),
        output='screen'
    )

    # 9. Delayed Launch Handler for DLIO and RViz
    # Ensures DLIO only starts IMU gravity calibration 1.5s AFTER the robot is spawned
    # and joint controllers are active and the robot is completely motionless on the ground.
    delayed_dlio_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_jpc,
            on_exit=[
                TimerAction(
                    period=1.5,
                    actions=[dlio_odom_node, elevation_map_node, dlio_map_node, rviz]
                )
            ]
        )
    )

    return LaunchDescription([
        gui_arg,
        dlio_arg,
        mapping_arg,
        rviz_arg,
        gazebo,
        robot_state_publisher,
        delayed_spawn_entity,
        spawn_jsb,
        spawn_jpc,
        rl_controller,
        delayed_dlio_handler
    ])
