module covenant;
[__AttributeUsage(_AttributeTargets.Var)] public struct FrameAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)] public struct UnitAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)] public struct ColorAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)] public struct MapsAttribute { string from; string to; };
[__AttributeUsage(_AttributeTargets.Var)] public struct ConventionAttribute { string kv; };
