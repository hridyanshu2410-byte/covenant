"""Image file I/O signatures: what encoding the pixels handed to (or returned
by) an image library are in.

* stb_image 2.x (stb_image.h header docs): stbi_load* return the file's 8/16-bit
  code values unchanged (PNG / JPEG / BMP / TGA are display-referred: sRGB);
  stbi_loadf* convert LDR files to linear with stbi_ldr_to_hdr_gamma (2.2)
  and return HDR (.hdr) data as stored (linear).
* stb_image_write 1.x: stbi_write_png / bmp / tga / jpg write 8-bit code values
  as given; the formats are display-referred, so the data must be sRGB
  encoded. stbi_write_hdr takes linear float.
* libpng 1.6 (PNG spec, ISO/IEC 15948 / W3C PNG 3rd ed. 11.3.3): samples are
  stored as given; without gAMA / sRGB / iCCP / cICP a decoder assumes sRGB.
  png_set_sRGB* declares sRGB; png_set_gAMA(g) declares a power-law encoding
  with exponent g (0.45455 = sRGB-like); png_set_gAMA(1.0) declares linear.
* lodepng 2023: lodepng_encode*_file / lodepng::encode write 8/16-bit code
  values; no gAMA chunk unless info_png.color has one, so sRGB is assumed.
* tinyexr 1.0: SaveEXR / SaveEXRImageToFile take linear scene-referred float
  (OpenEXR stores linear light).
"""
SRGB, LINEAR = "srgb", "linear"

# function -> (direction, encoding of the pixel data, bit depth class)
SINKS = {
    "stbi_write_png": SRGB, "stbi_write_bmp": SRGB, "stbi_write_tga": SRGB,
    "stbi_write_jpg": SRGB, "stbi_write_png_to_func": SRGB, "stbi_write_jpg_to_func": SRGB,
    "stbi_write_hdr": LINEAR,
    "png_write_image": SRGB, "png_write_row": SRGB, "png_write_rows": SRGB,
    "png_image_write_to_file": SRGB, "png_image_write_to_memory": SRGB,
    "png_image_write_to_stdio": SRGB,
    "lodepng_encode32_file": SRGB, "lodepng_encode24_file": SRGB, "lodepng_encode_file": SRGB,
    "lodepng_encode_memory": SRGB, "lodepng_encode32": SRGB, "lodepng_encode24": SRGB,
    "SaveEXR": LINEAR, "SaveEXRImageToFile": LINEAR, "SaveEXRToMemory": LINEAR,
}
SOURCES = {
    "stbi_load": SRGB, "stbi_load_from_memory": SRGB, "stbi_load_16": SRGB,
    "stbi_loadf": LINEAR, "stbi_loadf_from_memory": LINEAR,
    "lodepng_decode32_file": SRGB, "lodepng_decode24_file": SRGB,
    "png_read_image": SRGB, "png_image_finish_read": SRGB,
    "LoadEXR": LINEAR, "LoadEXRFromMemory": LINEAR,
}


def png_declared(gamma=None, srgb_intent=None):
    """Encoding a PNG declares through its chunks (None = not declared:
    readers assume sRGB)."""
    if srgb_intent is not None:
        return SRGB
    if gamma is not None:
        if abs(gamma - 1.0) < 0.02:
            return LINEAR
        if abs(gamma - 1 / 2.2) < 0.02:
            return SRGB
        return f"gamma {1 / gamma:.2f}"
    return None


def sink_encoding(fn, declared=None):
    """Encoding the pixel data handed to fn must have."""
    enc = SINKS.get(fn)
    if enc == SRGB and declared:
        return declared
    return enc
