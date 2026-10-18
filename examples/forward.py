"""A small forward renderer with async light culling, a tonemap pass and two
frames in flight, written against the Contract IR. Each `bugs` flag seeds one
realistic bug; with no flags the program is correct.

Bug flags
  srgb            albedo texture uses a UNORM format, so sRGB data is never decoded
  world_view      light vector computed from a view-space position and a
                  world-space light position
  double_encode   tonemap output encoded to sRGB by hand AND by the sRGB swapchain
  units_cm        host loads light positions authored in centimeters
  host_layout     hand-written C++ Light struct without std430 padding
  missing_bind    draw before the lights buffer is bound
  async_race      graphics reads the light grid without waiting for async compute
  in_flight       CPU rewrites the camera buffer without waiting for the frame fence
  no_barrier      HDR target sampled without the render-to-sample transition
"""
from covenant.ir import Contract, Field, Shader, Pipeline, HostWrite
from covenant.timing import Frame, Use, Barrier, Signal, Wait


def build(bugs=()):
    bugs = set(bugs)
    camera = Contract("camera", "uniform", fields=[
        Field("model", "mat4"), Field("view", "mat4"),
        Field("eye", "vec3", {"frame": "world", "unit": "m"})],
        matrix_frames={"model": ("object", "world"), "view": ("world", "view")},
        per_frame=2)
    lights = Contract("lights", "storage_buffer", fields=[
        Field("position", "vec3", {"frame": "world", "unit": "m"}),
        Field("color", "vec3", {"color": "linear"}),
        Field("radius", "float", {"unit": "m"})])
    albedo = Contract("albedo_tex", "texture", tag={"color": "srgb"},
                      format="rgba8_unorm" if "srgb" in bugs else "rgba8_srgb")
    hdr = Contract("hdr", "render_target", format="rgba16f")
    swap = Contract("swapchain", "render_target", format="bgra8_srgb")

    # ---- forward pass
    vs = Shader("forward.vert", "vertex")
    p = vs.input("position", "vec3", {"frame": "object", "unit": "m"})
    nrm = vs.input("normal", "vec3", {"frame": "object"})
    uv = vs.input("uv", "vec2")
    vs.output("world_pos", vs.transform(vs.read(camera, "model"), p, "world_pos = model * position"))
    vs.output("normal", vs.transform(vs.read(camera, "model"), nrm, "normal = model * normal"))
    vs.output("uv", uv)

    fs = Shader("forward.frag", "fragment")
    n_in = fs.input("normal", "vec3")          # declared in a different order
    wp = fs.input("world_pos", "vec3")         # than the vertex outputs
    uv_in = fs.input("uv", "vec2")
    if "world_view" in bugs:
        wp = fs.transform(fs.read(camera, "view"), wp, "p = view * world_pos")
    to_light = fs.sub(fs.read(lights, "position"), wp, "light.position - p")
    ndl = fs.dot(fs.normalize(n_in, "normalize(normal)"),
                 fs.normalize(to_light, "normalize(L)"), "dot(N, L)")
    base = fs.sample(fs.read(albedo), uv_in, "texture(albedo, uv)")
    lit = fs.scale(ndl, fs.modulate(base, fs.read(lights, "color"), "albedo * light.color"),
                   "ndl * (albedo * light.color)")
    fs.write(hdr, lit)
    forward = Pipeline("forward", [vs, fs])

    # ---- tonemap pass
    tv = Shader("tonemap.vert", "vertex")
    tv.output("uv", tv.input("uv", "vec2"))
    tf = Shader("tonemap.frag", "fragment")
    c = tf.sample(tf.read(hdr), tf.input("uv", "vec2"), "texture(hdr, uv)")
    t = tf.tonemap(c, "aces(c)")
    if "double_encode" in bugs:
        t = tf.linear_to_srgb(t, "linear_to_srgb(aces(c))")
    tf.write(swap, t)
    tonemap = Pipeline("tonemap", [tv, tf])

    host = [HostWrite(lights, {"position": {"unit": "cm" if "units_cm" in bugs else "m"}},
                      "scene_loader.cpp:88")]

    host_light = ([("position", "vec3"), ("color", "vec3"), ("radius", "float")]
                  if "host_layout" in bugs else None)

    commands = [("bind_pipeline", forward), ("bind", camera)]
    if "missing_bind" not in bugs:
        commands.append(("bind", lights))
    commands += [("bind", albedo), ("draw",)]

    frame = build_frame(bugs)
    return dict(pipelines=[forward, tonemap], host_writes=host, commands=commands,
                frame=frame,
                layouts=[(lights, host_light, "renderer.hpp:12")] if host_light else [],
                contracts=dict(camera=camera, lights=lights, albedo=albedo, hdr=hdr,
                               swap=swap))


def build_frame(bugs, frames=4, in_flight=2):
    fr = Frame(initial_layouts={"hdr": "shader_read", "swapchain": "present"})
    host, comp, gfx = fr.q("host"), fr.q("compute"), fr.q("graphics")
    for f in range(frames):
        k = f % in_flight
        if f >= in_flight and "in_flight" not in bugs:
            host.append(Wait(f"fence{f - in_flight}", f"frame {f}: wait for frame {f - in_flight}"))
        host.append(Use(f"camera[{k}]", "write", where=f"frame {f}: CPU updates camera[{k}]"))
        host.append(Signal(f"submit{f}", f"frame {f}: submit"))

        comp += [Wait(f"submit{f}"),
                 Use(f"light_grid[{k}]", "write", where=f"frame {f}: light culling (compute)"),
                 Signal(f"cull{f}")]

        gfx.append(Wait(f"submit{f}"))
        gfx.append(Barrier("swapchain", "present", "color_attachment", f"frame {f}: acquire"))
        gfx.append(Barrier("hdr", "shader_read", "color_attachment", f"frame {f}: hdr to target"))
        gfx.append(Use(f"camera[{k}]", "read", where=f"frame {f}: forward pass reads camera"))
        if "async_race" not in bugs:
            gfx.append(Wait(f"cull{f}", f"frame {f}: wait for culling"))
        gfx.append(Use(f"light_grid[{k}]", "read", where=f"frame {f}: forward pass reads light grid"))
        gfx.append(Use("hdr", "write", "color_attachment", f"frame {f}: forward pass draws"))
        if "no_barrier" not in bugs:
            gfx.append(Barrier("hdr", "color_attachment", "shader_read", f"frame {f}: hdr to sampled"))
        gfx.append(Use("hdr", "read", "shader_read", f"frame {f}: tonemap samples hdr"))
        gfx.append(Use("swapchain", "write", "color_attachment", f"frame {f}: tonemap writes"))
        gfx.append(Barrier("swapchain", "color_attachment", "present", f"frame {f}: present"))
        gfx.append(Signal(f"fence{f}", f"frame {f}: fence"))
    return fr
