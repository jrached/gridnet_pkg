from launch import LaunchDescription 
from launch_ros.actions import Node 
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    namespace_arg = DeclareLaunchArgument('namespace', default_value='PX05', description='Namespace of the nodes')
    cloud_topic_arg = DeclareLaunchArgument('cloud_topic', default_value="livox/lidar", description='Name of input point cloud topic')
    pose_topic_arg = DeclareLaunchArgument('pose_topic', default_value="world", description="Name of vehicle pose topic")
    twist_topic_arg = DeclareLaunchArgument('twist_topic', default_value="mocap/twist", description="Name of vehicle twist topic name")

    namespace = LaunchConfiguration('namespace')
    cloud_topic = LaunchConfiguration("cloud_topic")
    pose_topic = LaunchConfiguration("pose_topic")
    twist_topic = LaunchConfiguration("twist_topic")

    return LaunchDescription([
        namespace_arg, 
        cloud_topic_arg,
        pose_topic_arg,
        twist_topic_arg,
        Node(
            package='gridnet_pkg',
            namespace=namespace, 
            executable='gridnet_node',
            name='gridnet_node',
            remappings=[
                ('cloud_topic', cloud_topic),
                ('pose_topic',  pose_topic),
                ('twist_topic', twist_topic),
            ],
            prefix='xterm -e gdb -q -ex run --args python3', # gdb debugging
        )
    ])