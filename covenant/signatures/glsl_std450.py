"""GLSL.std.450 extended instructions (Khronos "GLSL.std.450" specification,
Version 1.00 Rev. 9) and their meaning signatures.

COLOR_SAME: the result has the operands' meaning (component-wise selection,
rounding, clamping, interpolation): one color / frame / unit class.
NONLINEAR: transfer-capable; their color label comes only from probe
fingerprinting (transfer.fit) or stays opaque.
ANGLE_IN: the operand is an angle in radians; ANGLE_OUT: the result is.
RADIANS / DEGREES: conversions (degrees -> radians is the pi/180 scale).
"""
# FSign is not here: sign(x) is -1, 0 or 1 whatever x encodes, so it carries
# none of x's meaning (sign(v) * pow(|v|, g) is the odd extension of a transfer
# function, as in WebGPU external-texture gamma parameters).
COLOR_SAME = {"FAbs", "Floor", "Ceil", "Fract", "Round", "RoundEven", "Trunc",
              "FMin", "FMax", "NMin", "NMax", "FClamp", "NClamp", "FMix"}
DIRECTION = {"Normalize", "Cross", "Reflect", "Refract", "FaceForward"}   # frame kept, unit dropped
LENGTH = {"Length", "Distance"}                                          # unit kept, frame dropped
NONLINEAR = {"Pow", "Exp", "Exp2", "Log", "Log2", "Sqrt", "InverseSqrt", "SmoothStep", "Step",
             "Sinh", "Cosh", "Tanh"}
ANGLE_IN = {"Sin", "Cos", "Tan"}
ANGLE_OUT = {"Asin", "Acos", "Atan", "Atan2"}
RADIANS, DEGREES = "Radians", "Degrees"
ANGLE_CONVERT = {RADIANS: -1, DEGREES: +1}          # angle element added (deg = +1)
MATRIX_INVERSE = "MatrixInverse"                    # swaps from / to
