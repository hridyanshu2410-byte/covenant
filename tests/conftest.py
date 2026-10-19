"""Test environment. The Vulkan-Headers include directory of the README
(COVENANT_VKINC) also supplies the flag names that elide.py reads; tests whose
tools are missing (libclang, slangc, spirv-val, a C++ compiler, the headers)
are skipped, not failed."""
import os

_inc = os.environ.get("COVENANT_VKINC")
if _inc and not os.environ.get("VULKAN_HEADER"):
    os.environ["VULKAN_HEADER"] = os.path.join(_inc, "vulkan", "vulkan_core.h")


def have_vulkan_header():
    from covenant import elide
    try:
        elide.VkNames.find_header()
        return True
    except FileNotFoundError:
        return False
