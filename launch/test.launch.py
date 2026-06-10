from launch import LaunchDescription 
from launch_ros.actions import Node 
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    namespace_arg = DeclareLaunchArgument('namespace', default_value='PX05', description='Namespace of the nodes')
    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        namespace_arg, 
        Node(
            package='gridnet_pkg',
            namespace=namespace, 
            executable='test_node',
            name='test_node'
        )
    ])