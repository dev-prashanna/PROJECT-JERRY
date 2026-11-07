from pathlib import Path
import os
import xml.etree.ElementTree as ET
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess,OpaqueFunction,RegisterEventHandler,EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def setup(context):
 p=Path(get_package_share_directory('vbmd_description'))
 headless=LaunchConfiguration('headless').perform(context).lower()=='true'
 world=LaunchConfiguration('world').perform(context)
 world_name=ET.parse(world).getroot().find('world').get('name')
 cmd=['gz','sim','-r','-v','2']
 if headless:cmd+=['-s','--headless-rendering']
 cmd+=[world]
 resource=str(p.parent)+os.pathsep+os.environ.get('GZ_SIM_RESOURCE_PATH','')
 gz=ExecuteProcess(cmd=cmd,output='screen',additional_env={'GZ_SIM_RESOURCE_PATH':resource,'GZ_PARTITION':os.environ.get('GZ_PARTITION','vbmd_dev'),'GZ_IP':os.environ.get('GZ_IP','127.0.0.1'),'QT_QPA_PLATFORM':'xcb'})
 common={'use_sim_time':True}
 return [RegisterEventHandler(OnProcessExit(target_action=gz,on_exit=[EmitEvent(event=Shutdown(reason='Gazebo closed; stopping its robot bridge'))])),gz,Node(package='ros_gz_bridge',executable='parameter_bridge',parameters=[{'config_file':str(p/'config/gazebo_bridge.yaml'),'use_sim_time':True}],output='screen',additional_env={'GZ_PARTITION':os.environ.get('GZ_PARTITION','vbmd_dev'),'GZ_IP':os.environ.get('GZ_IP','127.0.0.1')}),
 Node(package='ros_gz_bridge',executable='parameter_bridge',name='vbmd_world_control_bridge',arguments=['/world/'+world_name+'/control@ros_gz_interfaces/srv/ControlWorld'],output='screen',additional_env={'GZ_PARTITION':os.environ.get('GZ_PARTITION','vbmd_dev'),'GZ_IP':os.environ.get('GZ_IP','127.0.0.1')}),
 ExecuteProcess(cmd=['/usr/bin/python3',str(p/'scripts/gazebo_adapter.py'),'--ros-args','-p','use_sim_time:=true','-p','autonomy:='+LaunchConfiguration('autonomy').perform(context),'-p','protected_motion:='+LaunchConfiguration('protected_motion').perform(context)],output='screen'),
 Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':(p/'urdf/vbmd.urdf').read_text(),**common}],output='screen'),
 Node(package='rviz2',executable='rviz2',arguments=['-d',str(p/'config/vbmd.rviz')],parameters=[common],condition=IfCondition(LaunchConfiguration('rviz')))]
def generate_launch_description():
 p=Path(get_package_share_directory('vbmd_description'))
 return LaunchDescription([DeclareLaunchArgument('headless',default_value='false'),DeclareLaunchArgument('rviz',default_value='false'),DeclareLaunchArgument('autonomy',default_value='false'),DeclareLaunchArgument('protected_motion',default_value='false'),DeclareLaunchArgument('world',default_value=str(p/'worlds/vbmd_lab.sdf')),OpaqueFunction(function=setup)])
