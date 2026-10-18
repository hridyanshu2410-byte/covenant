/* COVENANT image-I/O interposer (libcovio.so), loaded with LD_PRELOAD next to
 * the model driver. It records what a program hands to image writers so the
 * meaning checker can fit the host's pixel path (covenant/witness.py):
 *
 *   fopen/fwrite/fclose          raw pixel dumps (first bytes of each write)
 *   libpng                       png_set_IHDR / gAMA / sRGB, png_write_row /
 *                                png_write_image, png_image_write_to_file
 *   lodepng                      lodepng_encode32_file / 24 / encode_file
 *   stb_image_write              stbi_write_png / stbi_write_hdr (when linked
 *                                from a shared library; a header-only copy
 *                                compiled into the program cannot be
 *                                interposed and is seen through fwrite only)
 *   tinyexr                      SaveEXR (same caveat)
 *
 * Records are JSON lines appended to $COVENANT_IO_TRACE with plain write(2),
 * so the interposer never re-enters stdio. Writes to $COVENANT_TRACE and
 * under $COVENANT_SPIRV_DIR are ignored.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HEAD_MAX (64 * 1024)
#define MAXF 256

static pthread_mutex_t mu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static int out_fd = -2;
static __thread int busy = 0;
static unsigned long long seq = 0;

static struct { FILE* f; char path[512]; long written; } files[MAXF];

static void out_open(void) {
  if (out_fd != -2) return;
  const char* p = getenv("COVENANT_IO_TRACE");
  out_fd = (p && *p) ? open(p, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644) : -1;
}

typedef struct { char* s; size_t n, cap; } buf_t;
static void put(buf_t* b, const char* s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    size_t c = b->cap ? b->cap * 2 : 4096;
    while (c < b->n + n + 1) c *= 2;
    b->s = realloc(b->s, c); b->cap = c;
  }
  memcpy(b->s + b->n, s, n); b->n += n; b->s[b->n] = 0;
}
static void putf(buf_t* b, const char* fmt, ...) {
  char tmp[1024]; va_list ap; va_start(ap, fmt);
  int n = vsnprintf(tmp, sizeof tmp, fmt, ap); va_end(ap);
  if (n > 0) put(b, tmp, (size_t)n < sizeof tmp ? (size_t)n : sizeof tmp - 1);
}
static void putstr(buf_t* b, const char* s) {
  put(b, "\"", 1);
  for (; s && *s; s++) {
    if (*s == '"' || *s == '\\') { put(b, "\\", 1); put(b, s, 1); }
    else if ((unsigned char)*s < 0x20) putf(b, "\\u%04x", *s);
    else put(b, s, 1);
  }
  put(b, "\"", 1);
}
static void puthex(buf_t* b, const void* data, size_t n) {
  static const char* H = "0123456789abcdef";
  const unsigned char* d = data;
  put(b, "\"", 1);
  char tmp[512]; size_t k = 0;
  for (size_t i = 0; i < n; i++) {
    tmp[k++] = H[d[i] >> 4]; tmp[k++] = H[d[i] & 15];
    if (k == sizeof tmp) { put(b, tmp, k); k = 0; }
  }
  put(b, tmp, k);
  put(b, "\"", 1);
}
static void begin(buf_t* b, const char* call) {
  b->s = NULL; b->n = b->cap = 0;
  putf(b, "{\"fn\":\"covenant_io\",\"io_seq\":%llu,\"call\":", ++seq);
  putstr(b, call);
}
static void finish(buf_t* b) {
  put(b, "}\n", 2);
  out_open();
  if (out_fd >= 0) { ssize_t r = write(out_fd, b->s, b->n); (void)r; }
  free(b->s);
}

static int ignored(const char* path) {
  if (!path) return 1;
  const char* t = getenv("COVENANT_TRACE");
  if (t && !strcmp(t, path)) return 1;
  const char* io = getenv("COVENANT_IO_TRACE");
  if (io && !strcmp(io, path)) return 1;
  const char* d = getenv("COVENANT_SPIRV_DIR");
  if (d && *d && !strncmp(d, path, strlen(d))) return 1;
  return 0;
}

#define REAL(name, type) static __typeof__(name)* real_##name; if (!real_##name) real_##name = (__typeof__(name)*)dlsym(RTLD_NEXT, #name)

/* ------------------------------------------------------------- stdio */
FILE* fopen(const char* path, const char* mode) {
  REAL(fopen, FILE* (*)(const char*, const char*));
  FILE* f = real_fopen(path, mode);
  if (f && !busy && mode && (strchr(mode, 'w') || strchr(mode, 'a')) && !ignored(path)) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < MAXF; i++)
      if (!files[i].f) { files[i].f = f; snprintf(files[i].path, sizeof files[i].path, "%s", path); files[i].written = 0; break; }
    pthread_mutex_unlock(&mu);
  }
  return f;
}
FILE* fopen64(const char* path, const char* mode) { return fopen(path, mode); }

size_t fwrite(const void* p, size_t sz, size_t n, FILE* f) {
  REAL(fwrite, size_t (*)(const void*, size_t, size_t, FILE*));
  size_t r = real_fwrite(p, sz, n, f);
  if (busy || !f) return r;
  pthread_mutex_lock(&mu);
  for (int i = 0; i < MAXF; i++) {
    if (files[i].f != f) continue;
    busy = 1;
    size_t bytes = sz * n;
    buf_t b; begin(&b, "fwrite");
    put(&b, ",\"path\":", 8); putstr(&b, files[i].path);
    putf(&b, ",\"pos\":%ld,\"size\":%zu,\"head\":", files[i].written, bytes);
    puthex(&b, p, bytes < HEAD_MAX ? bytes : HEAD_MAX);
    finish(&b);
    files[i].written += (long)bytes;
    busy = 0;
    break;
  }
  pthread_mutex_unlock(&mu);
  return r;
}

int fclose(FILE* f) {
  REAL(fclose, int (*)(FILE*));
  pthread_mutex_lock(&mu);
  for (int i = 0; i < MAXF; i++)
    if (files[i].f == f) {
      if (!busy) {
        busy = 1;
        buf_t b; begin(&b, "fclose");
        put(&b, ",\"path\":", 8); putstr(&b, files[i].path);
        putf(&b, ",\"size\":%ld", files[i].written);
        finish(&b);
        busy = 0;
      }
      files[i].f = NULL;
    }
  pthread_mutex_unlock(&mu);
  return real_fclose(f);
}

/* ------------------------------------------------------------- libpng */
static struct { void* png; unsigned w, h; int depth, ctype; } pngs[64];
static int png_slot(void* png) {
  for (int i = 0; i < 64; i++) if (pngs[i].png == png) return i;
  for (int i = 0; i < 64; i++) if (!pngs[i].png) { pngs[i].png = png; return i; }
  return 0;
}
static size_t png_rowbytes(int s) {
  static const int ch[7] = {1, 0, 3, 1, 2, 0, 4};   /* gray, -, rgb, palette, gray+a, -, rgba */
  int c = pngs[s].ctype >= 0 && pngs[s].ctype < 7 ? ch[pngs[s].ctype] : 4;
  return (size_t)pngs[s].w * (size_t)(c ? c : 4) * (size_t)(pngs[s].depth > 8 ? 2 : 1);
}

void png_set_IHDR(void* png, void* info, uint32_t w, uint32_t h, int depth, int ctype, int il, int comp, int filt) {
  REAL(png_set_IHDR, void (*)(void*, void*, uint32_t, uint32_t, int, int, int, int, int));
  pthread_mutex_lock(&mu);
  int s = png_slot(png); pngs[s].w = w; pngs[s].h = h; pngs[s].depth = depth; pngs[s].ctype = ctype;
  buf_t b; begin(&b, "png_set_IHDR");
  putf(&b, ",\"png\":\"%p\",\"width\":%u,\"height\":%u,\"bit_depth\":%d,\"color_type\":%d", png, w, h, depth, ctype);
  finish(&b);
  pthread_mutex_unlock(&mu);
  if (real_png_set_IHDR) real_png_set_IHDR(png, info, w, h, depth, ctype, il, comp, filt);
}
void png_set_gAMA(void* png, void* info, double g) {
  REAL(png_set_gAMA, void (*)(void*, void*, double));
  buf_t b; begin(&b, "png_set_gAMA"); putf(&b, ",\"png\":\"%p\",\"gamma\":%.6f", png, g); finish(&b);
  if (real_png_set_gAMA) real_png_set_gAMA(png, info, g);
}
void png_set_sRGB(void* png, void* info, int intent) {
  REAL(png_set_sRGB, void (*)(void*, void*, int));
  buf_t b; begin(&b, "png_set_sRGB"); putf(&b, ",\"png\":\"%p\",\"intent\":%d", png, intent); finish(&b);
  if (real_png_set_sRGB) real_png_set_sRGB(png, info, intent);
}
void png_set_sRGB_gAMA_and_cHRM(void* png, void* info, int intent) {
  REAL(png_set_sRGB_gAMA_and_cHRM, void (*)(void*, void*, int));
  buf_t b; begin(&b, "png_set_sRGB"); putf(&b, ",\"png\":\"%p\",\"intent\":%d", png, intent); finish(&b);
  if (real_png_set_sRGB_gAMA_and_cHRM) real_png_set_sRGB_gAMA_and_cHRM(png, info, intent);
}
static void log_row(void* png, const void* row, unsigned k) {
  pthread_mutex_lock(&mu);
  int s = png_slot(png);
  size_t n = png_rowbytes(s);
  buf_t b; begin(&b, "png_write_row");
  putf(&b, ",\"png\":\"%p\",\"row\":%u,\"width\":%u,\"height\":%u,\"bit_depth\":%d,\"color_type\":%d,\"data\":",
       png, k, pngs[s].w, pngs[s].h, pngs[s].depth, pngs[s].ctype);
  puthex(&b, row, n < HEAD_MAX ? n : HEAD_MAX);
  finish(&b);
  pthread_mutex_unlock(&mu);
}
static unsigned png_rows_logged[64];
void png_write_row(void* png, const void* row) {
  REAL(png_write_row, void (*)(void*, const void*));
  int s = png_slot(png);
  if (png_rows_logged[s] < 2) log_row(png, row, png_rows_logged[s]);
  png_rows_logged[s]++;
  if (real_png_write_row) real_png_write_row(png, row);
}
void png_write_image(void* png, void** rows) {
  REAL(png_write_image, void (*)(void*, void**));
  if (rows && rows[0]) log_row(png, rows[0], 0);
  if (real_png_write_image) real_png_write_image(png, rows);
}
int png_image_write_to_file(void* image, const char* file, int convert_to_8bit, const void* buffer,
                            int32_t row_stride, const void* colormap) {
  REAL(png_image_write_to_file, int (*)(void*, const char*, int, const void*, int32_t, const void*));
  /* png_image: opaque, version, width, height, format, flags, colormap_entries ... (png.h 1.6) */
  struct { void* opaque; uint32_t version, width, height, format; } *im = image;
  buf_t b; begin(&b, "png_image_write_to_file");
  put(&b, ",\"path\":", 8); putstr(&b, file);
  putf(&b, ",\"width\":%u,\"height\":%u,\"format\":%u,\"convert_to_8bit\":%d,\"data\":",
       im->width, im->height, im->format, convert_to_8bit);
  size_t ch = (im->format & 2 ? 3 : 1) + (im->format & 1 ? 1 : 0);   /* COLOR 0x2, ALPHA 0x1 */
  size_t bytes = (size_t)im->width * ch * ((im->format & 4) ? 2 : 1);  /* LINEAR: 16-bit */
  puthex(&b, buffer, bytes < HEAD_MAX ? bytes : HEAD_MAX);
  finish(&b);
  return real_png_image_write_to_file ? real_png_image_write_to_file(image, file, convert_to_8bit, buffer, row_stride, colormap) : 0;
}

/* ------------------------------------------------------------- lodepng */
static void log_pixels(const char* call, const char* file, const void* image, unsigned w, unsigned h, int ch, int bits) {
  buf_t b; begin(&b, call);
  put(&b, ",\"path\":", 8); putstr(&b, file);
  putf(&b, ",\"width\":%u,\"height\":%u,\"channels\":%d,\"bit_depth\":%d,\"data\":", w, h, ch, bits);
  size_t n = (size_t)w * ch * (bits > 8 ? bits / 8 : 1);
  puthex(&b, image, n < HEAD_MAX ? n : HEAD_MAX);
  finish(&b);
}
unsigned lodepng_encode32_file(const char* file, const unsigned char* image, unsigned w, unsigned h) {
  REAL(lodepng_encode32_file, unsigned (*)(const char*, const unsigned char*, unsigned, unsigned));
  log_pixels("lodepng_encode32_file", file, image, w, h, 4, 8);
  return real_lodepng_encode32_file ? real_lodepng_encode32_file(file, image, w, h) : 1;
}
unsigned lodepng_encode24_file(const char* file, const unsigned char* image, unsigned w, unsigned h) {
  REAL(lodepng_encode24_file, unsigned (*)(const char*, const unsigned char*, unsigned, unsigned));
  log_pixels("lodepng_encode24_file", file, image, w, h, 3, 8);
  return real_lodepng_encode24_file ? real_lodepng_encode24_file(file, image, w, h) : 1;
}

/* ------------------------------------------------------------- stb / tinyexr */
int stbi_write_png(const char* file, int w, int h, int comp, const void* data, int stride) {
  REAL(stbi_write_png, int (*)(const char*, int, int, int, const void*, int));
  log_pixels("stbi_write_png", file, data, (unsigned)w, (unsigned)h, comp, 8);
  return real_stbi_write_png ? real_stbi_write_png(file, w, h, comp, data, stride) : 0;
}
int stbi_write_hdr(const char* file, int w, int h, int comp, const float* data) {
  REAL(stbi_write_hdr, int (*)(const char*, int, int, int, const float*));
  log_pixels("stbi_write_hdr", file, data, (unsigned)w, (unsigned)h, comp, 32);
  return real_stbi_write_hdr ? real_stbi_write_hdr(file, w, h, comp, data) : 0;
}
int SaveEXR(const float* data, int w, int h, int comp, int fp16, const char* file, const char** err) {
  REAL(SaveEXR, int (*)(const float*, int, int, int, int, const char*, const char**));
  log_pixels("SaveEXR", file, data, (unsigned)w, (unsigned)h, comp, 32);
  return real_SaveEXR ? real_SaveEXR(data, w, h, comp, fp16, file, err) : -1;
}
