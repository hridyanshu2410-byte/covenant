# Cross-platform build of the barrier-elision workloads and VK_LAYER_COVENANT_elide.
# Linux:   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# Windows: cmake -S . -B build && cmake --build build --config Release      (Vulkan SDK installed)
# Outputs: build/[Release/]{particles,deferred,postchain}[.exe] and build/layer/[Release/]
#          {libVkLayer_COVENANT_elide.so | VkLayer_COVENANT_elide.dll} + VkLayer_COVENANT_elide.json
cmake_minimum_required(VERSION 3.16)
project(covenant_elide CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Vulkan REQUIRED)

foreach(w particles deferred postchain)
  add_executable(${w} ${w}.cpp)
  target_link_libraries(${w} PRIVATE Vulkan::Vulkan)
  if(MSVC)
    target_compile_definitions(${w} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  endif()
endforeach()

set(LAYER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../elide_layer/covenant_elide_layer.cpp)
add_library(VkLayer_COVENANT_elide SHARED ${LAYER_SRC})
target_include_directories(VkLayer_COVENANT_elide PRIVATE ${Vulkan_INCLUDE_DIRS})
set_target_properties(VkLayer_COVENANT_elide PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/layer
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/layer)
if(MSVC)
  target_compile_definitions(VkLayer_COVENANT_elide PRIVATE _CRT_SECURE_NO_WARNINGS)
  set(LAYER_LIB_PATH ".\\\\VkLayer_COVENANT_elide.dll")
elseif(WIN32)
  # MinGW-w64 (GCC) on Windows: the DLL keeps CMake's "lib" prefix.
  set(LAYER_LIB_PATH ".\\\\libVkLayer_COVENANT_elide.dll")
  target_link_options(VkLayer_COVENANT_elide PRIVATE -static-libgcc -static-libstdc++)
  foreach(w particles deferred postchain)
    target_link_options(${w} PRIVATE -static-libgcc -static-libstdc++)
  endforeach()
else()
  target_link_options(VkLayer_COVENANT_elide PRIVATE -Wl,-Bsymbolic)
  set(LAYER_LIB_PATH "./libVkLayer_COVENANT_elide.so")
endif()
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../../elide_layer/VkLayer_COVENANT_elide.json.in MANIFEST)
string(REPLACE "@LIB@" "${LAYER_LIB_PATH}" MANIFEST "${MANIFEST}")
file(GENERATE OUTPUT $<TARGET_FILE_DIR:VkLayer_COVENANT_elide>/VkLayer_COVENANT_elide.json CONTENT "${MANIFEST}")
