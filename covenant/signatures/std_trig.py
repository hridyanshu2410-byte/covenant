"""C / C++ <cmath> (ISO C 7.12.4, ISO C++ [c.math]): trigonometric functions take
their argument in radians; the inverse functions return radians."""
ANGLE_IN = {"sin", "cos", "tan", "sinf", "cosf", "tanf", "sinl", "cosl", "tanl",
            "sincos", "sincosf"}
ANGLE_OUT = {"asin", "acos", "atan", "atan2", "asinf", "acosf", "atanf", "atan2f"}
# multiplying by pi / 180 converts degrees -> radians (angle element -1);
# by 180 / pi the other way. Other constants leave the angle element alone.
import math as _m
DEG_TO_RAD = _m.pi / 180.0
RAD_TO_DEG = 180.0 / _m.pi


def scale_element(k, rel=1e-4):
    if k is None:
        return 0
    if abs(k - DEG_TO_RAD) <= rel * DEG_TO_RAD:
        return -1
    if abs(k - RAD_TO_DEG) <= rel * RAD_TO_DEG:
        return +1
    return 0
