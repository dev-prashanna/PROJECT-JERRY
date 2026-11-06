# Keep native Gazebo/Qt processes from loading incompatible VS Code Snap GTK libs.
# Source after the ROS setup; retain ROS/Gazebo's native LD_LIBRARY_PATH.
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
for vbmd_gui_variable in GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GIO_EXTRA_MODULES; do
  case "${!vbmd_gui_variable:-}" in
    *'/snap/'*) unset "$vbmd_gui_variable" ;;
  esac
done
unset vbmd_gui_variable
