cmake_minimum_required(VERSION 3.8)
project(jerry_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotionCommand.msg"
  "msg/NavigationState.msg"
  "msg/TaskState.msg"
  "msg/MemoryRecord.msg"
  "srv/Audio.srv"
  "srv/Ask.srv"
  "srv/Authority.srv"
  "srv/SubmitTask.srv"
  "srv/Journal.srv"
  "action/Mission.action"
  DEPENDENCIES builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
