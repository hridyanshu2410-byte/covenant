// COVENANT host annotations. Expands to a clang annotate attribute that the
// checker reads; compilers ignore it otherwise (GCC warns at most).
#pragma once
#if defined(__clang__)
#define COV(s) __attribute__((annotate("covenant:" s)))
#else
#define COV(s)
#endif
