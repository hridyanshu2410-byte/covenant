# Build the COVENANT model driver (libcovvk.so) and the triangle example.
# make            -> build/libcovvk.so (+ build/libvulkan.so{,.1}, copies for LD_LIBRARY_PATH use)
# make example    -> build/triangle, build/tri.vert.spv, build/tri.frag.spv
# make hostcapture -> build/hostcapture (host-write capture probe)
ifeq ($(origin CXX),default)
CXX := $(shell command -v clang++-20 >/dev/null 2>&1 && echo clang++-20 || echo g++)
endif
VKH ?= /usr/include/vulkan/vulkan_core.h
# VKINC: include dir of newer Vulkan-Headers (then VKH=$(VKINC)/vulkan/vulkan_core.h)
VKINC ?=
CXXFLAGS ?= -O1 -g
CXXFLAGS += -std=c++17 -fPIC -Wall -Wno-unused-function -Wno-unused-parameter -Wno-sign-compare
B := build

all: $(B)/libcovvk.so $(B)/libcovio.so

$(B)/covvk_enums.inc $(B)/covvk_stubs.inc: gen.py covvk.cpp $(VKH) | $(B)
	python3 gen.py $(VKH) covvk.cpp $(B)/covvk_enums.inc $(B)/covvk_stubs.inc

# only vk* symbols are exported (covvk.map)
$(B)/libcovvk.so: covvk.cpp covvk.map $(B)/covvk_enums.inc $(B)/covvk_stubs.inc
	$(CXX) $(CXXFLAGS) $(if $(VKINC),-I$(VKINC)) -I$(B) -shared -o $@ covvk.cpp -Wl,-soname,libcovvk.so -Wl,--version-script=covvk.map
	cp -f $@ $(B)/libvulkan.so.1   # copies, not symlinks (works on file systems without them)
	cp -f $@ $(B)/libvulkan.so

example: $(B)/triangle $(B)/tri.vert.spv $(B)/tri.frag.spv

$(B)/triangle: examples/triangle.cpp $(B)/libcovvk.so
	$(CXX) -std=c++17 -O1 -g -Wall -o $@ examples/triangle.cpp -L$(B) -lcovvk -Wl,-rpath,'$$ORIGIN'

# host-write capture probe (tests/test_strict_undefined_and_capture.py)
hostcapture: $(B)/hostcapture
$(B)/hostcapture: examples/hostcapture.cpp $(B)/libcovvk.so
	$(CXX) -std=c++17 -O1 -g -Wall $(if $(VKINC),-I$(VKINC)) -o $@ examples/hostcapture.cpp -L$(B) -lcovvk -lpthread -Wl,-rpath,'$$ORIGIN'

$(B)/tri.vert.spv: examples/tri.vert | $(B)
	glslangValidator -V -o $@ $<
$(B)/tri.frag.spv: examples/tri.frag | $(B)
	glslangValidator -V -o $@ $<

$(B):
	mkdir -p $(B)

clean:
	rm -rf $(B)

.PHONY: all example clean io hostcapture

# image-I/O interposer (LD_PRELOAD) for the meaning checker's host pixel path
io: $(B)/libcovio.so
$(B)/libcovio.so: covio.c | $(B)
	$(CC) -O1 -g -fPIC -shared -Wall -o $@ covio.c -ldl -lpthread
