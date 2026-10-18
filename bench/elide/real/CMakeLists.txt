# Layers for barrier elision on real samples (bench/elide/REAL-SAMPLES.md): VK_LAYER_COVENANT_elide and
# VK_LAYER_COVENANT_probe, both into build/layers/[Release/] together with their manifests.
# Linux:          cmake -S . -B build && cmake --build build -j
# Windows (MSVC): cmake -S . -B build && cmake --build build --config Release      (Vulkan SDK installed)
# Windows (MinGW-w64, GCC, from an MSYS2 UCRT64 or w64devkit shell):
#                 cmake -S . -B build -G "MinGW Makefiles" && cmake --build build -j
# See ../REAL-SAMPLES.md.
cmake_minimum_required(VERSION 3.16)
project(covenant_elide_real CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Vulkan REQUIRED)

set(ELIDE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../elide_layer)
set(PROBE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/probe_layer)

function(covenant_layer target src manifest_in)
  add_library(${target} SHARED ${src})
  target_include_directories(${target} PRIVATE ${Vulkan_INCLUDE_DIRS})
  set_target_properties(${target} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/layers
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/layers)
  if(WIN32)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  if(MSVC)
    set(lib ".\\\\${target}.dll")
  elseif(WIN32)
    # MinGW-w64: CMake keeps the "lib" prefix; link the GCC runtimes statically so the DLL
    # loads in any process (the samples may be MSVC builds).
    target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++ -static)
    set(lib ".\\\\lib${target}.dll")
  else()
    target_link_options(${target} PRIVATE -Wl,-Bsymbolic)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    set(lib "./lib${target}.so")
  endif()
  file(READ ${manifest_in} m)
  string(REPLACE "@LIB@" "${lib}" m "${m}")
  file(GENERATE OUTPUT $<TARGET_FILE_DIR:${target}>/${target}.json CONTENT "${m}")
endfunction()

covenant_layer(VkLayer_COVENANT_elide ${ELIDE_SRC}/covenant_elide_layer.cpp ${ELIDE_SRC}/VkLayer_COVENANT_elide.json.in)
covenant_layer(VkLayer_COVENANT_probe ${PROBE_SRC}/covenant_probe_layer.cpp ${PROBE_SRC}/VkLayer_COVENANT_probe.json.in)
