from pathlib import Path
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess,OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def setup(context):
 pkg=Path(get_package_share_directory('vbmd_description'))
 py=LaunchConfiguration('python').perform(context)
 args=[py,str(pkg/'scripts/run_sim.py'),'--ros']
 if LaunchConfiguration('headless').perform(context).lower()=='true':args+=['--headless']
 if LaunchConfiguration('ground_truth').perform(context).lower()=='true':args+=['--ground-truth']
 if LaunchConfiguration('autonomy').perform(context).lower()=='true':args+=['--autonomy']
 return [ExecuteProcess(cmd=args,output='screen'),Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':(pkg/'urdf/vbmd.urdf').read_text(),'use_sim_time':True}],output='screen'),Node(package='rviz2',executable='rviz2',arguments=['-d',str(pkg/'config/vbmd.rviz')],parameters=[{'use_sim_time':True}],condition=IfCondition(LaunchConfiguration('rviz')))]
def generate_launch_description():
 return LaunchDescription([
 DeclareLaunchArgument('python',default_value=os.environ.get('VBMD_SIM_PYTHON',str(Path.cwd()/'.venv-vbmd-sim/bin/python'))),
 DeclareLaunchArgument('headless',default_value='false'),DeclareLaunchArgument('rviz',default_value='false'),
 DeclareLaunchArgument('ground_truth',default_value='false'),DeclareLaunchArgument('autonomy',default_value='false'),OpaqueFunction(function=setup)])
