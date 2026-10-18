# Program table for the repair cases (sourced). prog <prog> <source tree> sets WD, EXE, ARGS, ENVS, XLIB.
# <source tree> is the tree of one variant (original or patched), built by the build_*.sh scripts.
# GStreamer: the gst-plugins-bad Vulkan library and plugin come from the tree; every other plugin from
# the system ($GST_SYSTEM_PLUGINS, default /usr/lib/x86_64-linux-gnu/gstreamer-1.0), through a private
# plugin directory of symlinks and a private registry.
prog() {
  local p=$1 S=$2; ENVS=""; WD=$WORK; XLIB=""
  case $p in
    vkmark_*) local b=${p#vkmark_}; EXE=$S/build/src/vkmark
      case $b in shading) b="shading:shading=phong";; esac
      ARGS="--winsys-dir $S/build/src --data-dir $S/data -b $b:duration=600";;
    imgui_glfw|imgui_sdl2) WD=$S/examples/example_${p#imgui_}_vulkan; EXE=./example_${p#imgui_}_vulkan; ARGS="";;
    gst_*)
      local PD=$S/build/plugin-dir SYS=${GST_SYSTEM_PLUGINS:-/usr/lib/x86_64-linux-gnu/gstreamer-1.0}; mkdir -p "$PD"
      for f in "$SYS"/*.so; do [ "$(basename "$f")" = libgstvulkan.so ] || ln -sf "$f" "$PD/"; done
      ln -sf "$S/build/ext/vulkan/libgstvulkan.so" "$PD/libgstvulkan.so"
      XLIB=$S/build/gst-libs/gst/vulkan
      ENVS="GST_PLUGIN_SYSTEM_PATH_1_0=$PD GST_PLUGIN_PATH_1_0= GST_REGISTRY_1_0=$S/build/registry.bin GST_REGISTRY_FORK=no"
      EXE=gst-launch-1.0; ARGS=$(gst_args "${p#gst_}") || return 1;;
    sdl_*) WD=$S/build/test; EXE=./${p#sdl_}; ARGS=""; ENVS="SDL_VIDEO_DRIVER=x11 SDL_RENDER_DRIVER=vulkan";;
    *) echo "unknown program $p" >&2; return 1;;
  esac
}
gst_args() {
  case $1 in
    sink) echo "videotestsrc num-buffers=60 ! vulkanupload ! vulkansink";;
    colorconvert) echo "videotestsrc num-buffers=60 ! video/x-raw,format=NV12 ! vulkanupload ! vulkancolorconvert ! vulkansink";;
    overlay) echo "videotestsrc num-buffers=60 ! textoverlay text=covenant ! vulkanupload ! vulkanoverlaycompositor ! vulkansink";;
    viewconvert) echo "videotestsrc num-buffers=60 ! vulkanupload ! vulkanviewconvert ! vulkansink";;
    shaderspv) echo "videotestsrc num-buffers=60 ! vulkanupload ! vulkanshaderspv ! vulkansink";;
    *) return 1;;
  esac
}
