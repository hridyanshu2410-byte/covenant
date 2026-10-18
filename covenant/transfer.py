"""Transfer-function library and probe fitting.

A probe evaluates a shader (or host) slice y = f(x) on a ramp and fits it
against this library. Every entry is written from its specification:

  sRGB      IEC 61966-2-1 (piecewise; gamma 2.2 / 2.4 / BT.1886 are the same
            class for conflict purposes: they are display-referred SDR curves)
  PQ        SMPTE ST 2084 (input normalised to 10000 cd/m^2)
  HLG       ITU-R BT.2100 hybrid log-gamma OETF
  tonemaps  linear HDR -> linear display light (Reinhard, exponential,
            ACES fitted (Narkowicz 2015), Uncharted 2 / Hable): identity class
            for the transfer dimension (they keep values linear)
  affine    a * x + b: identity class

fit(xs, ys) returns a Fit with the group element (a reduced FreeGroup word:
(("srgb", +1),) = sRGB encode) only when the best class is unambiguous
and the curve is monotone; otherwise None (an opaque edge: no constraint, no
alarm).
"""
import math
from dataclasses import dataclass

import numpy as np

PQ_M1, PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
PQ_C1, PQ_C2, PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073


def srgb_enc(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_dec(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def gamma_enc(g):
    return lambda x: np.power(np.clip(x, 0, None), 1 / g)


def gamma_dec(g):
    return lambda x: np.power(np.clip(x, 0, None), g)


def pq_enc(x):
    y = np.power(np.clip(x, 0, None), PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * y) / (1 + PQ_C3 * y), PQ_M2)


def pq_dec(x):
    e = np.power(np.clip(x, 0, None), 1 / PQ_M2)
    return np.power(np.clip(e - PQ_C1, 0, None) / (PQ_C2 - PQ_C3 * e), 1 / PQ_M1)


def hlg_enc(x):
    x = np.clip(x, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x <= 1 / 12, np.sqrt(3 * x), HLG_A * np.log(np.maximum(12 * x - HLG_B, 1e-12)) + HLG_C)


def hlg_dec(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.5, x * x / 3, (np.exp((x - HLG_C) / HLG_A) + HLG_B) / 12)


def reinhard(x):
    return x / (1 + x)


def tm_exp(x):
    return 1 - np.exp(-x)


def aces_fit(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def _hable(x):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def hable(x):
    return _hable(x) / _hable(11.2)


# (name, class, group element, function, input-scale grid kind)
ENC, DEC = 1, -1
LIBRARY = [
    ("sRGB encode", "srgb", (("srgb", ENC),), srgb_enc, "fixed"),
    ("sRGB decode", "srgb", (("srgb", DEC),), srgb_dec, "fixed"),
    ("gamma 2.2 encode", "srgb", (("srgb", ENC),), gamma_enc(2.2), "fixed"),
    ("gamma 2.2 decode", "srgb", (("srgb", DEC),), gamma_dec(2.2), "fixed"),
    ("gamma 2.4 encode", "srgb", (("srgb", ENC),), gamma_enc(2.4), "fixed"),
    ("gamma 2.4 decode", "srgb", (("srgb", DEC),), gamma_dec(2.4), "fixed"),
    ("gamma 2.0 encode", "srgb", (("srgb", ENC),), gamma_enc(2.0), "fixed"),
    ("gamma 2.0 decode", "srgb", (("srgb", DEC),), gamma_dec(2.0), "fixed"),
    ("PQ (ST 2084) encode", "pq", (("pq", ENC),), pq_enc, "pq"),
    ("PQ (ST 2084) decode", "pq", (("pq", DEC),), pq_dec, "out"),
    ("HLG encode", "hlg", (("hlg", ENC),), hlg_enc, "fixed"),
    ("HLG decode", "hlg", (("hlg", DEC),), hlg_dec, "fixed"),
    ("Reinhard tonemap", "tonemap", (), reinhard, "wide"),
    ("exponential tonemap", "tonemap", (), tm_exp, "wide"),
    ("ACES fitted tonemap", "tonemap", (), aces_fit, "wide"),
    ("Hable tonemap", "tonemap", (), hable, "wide"),
]
PRETTY = {"srgb": "sRGB", "pq": "PQ", "hlg": "HLG", "rgbp": "RGBP"}

# PQ input: peak luminance between 10 and 10000 cd/m^2 (scale 1e-3 .. 1); HLG and
# the SDR curves take normalised [0, 1] signals (no input scale)
SCALES = {"fixed": [1.0], "pq": list(np.logspace(-3, 0, 31)),
          "wide": list(np.logspace(-4, 2, 121)), "out": [1.0]}
# log-spaced low end (power laws have unbounded slope at 0, tonemaps do not)
# plus an even spread over the rest of [0, 1]
RAMP = np.concatenate([np.logspace(-3, -1, 16, endpoint=False), np.linspace(0.1, 1.0, 32)])

TOL = 0.006            # accepted normalised RMS error (1.5% let low-scale PQ pass as sRGB)
MARGIN = 2.5           # the runner-up class must be this much worse


@dataclass
class Fit:
    name: str            # library entry, e.g. "PQ (ST 2084) encode"
    cls: str             # "srgb", "pq", "hlg", "tonemap", "affine"
    element: tuple       # FreeGroup word
    err: float
    scale: float = 1.0
    runner_up: str = ""

    def describe(self):
        s = f" of {self.scale:.3g} x" if self.scale not in (1.0,) and self.cls in ("pq", "tonemap", "hlg") else ""
        return f"{self.name}{s} (fit error {self.err:.1%})"


def _nrms(y, yh):
    rng = float(np.max(y) - np.min(y))
    if rng <= 1e-12:
        return float("inf")
    return float(np.sqrt(np.mean((y - yh) ** 2)) / rng)


def _fit_one(xs, ys, fn, grid):
    best = (float("inf"), 1.0)
    for s in SCALES[grid]:
        with np.errstate(all="ignore"):
            t = fn(xs * s)
        if not np.all(np.isfinite(t)):
            continue
        tt = float(np.dot(t, t))
        if tt <= 1e-20:
            continue
        a = float(np.dot(ys, t)) / tt          # output scale, no offset
        if a <= 0:
            continue
        e = _nrms(ys, a * t)
        if e < best[0]:
            best = (e, s)
    return best


def _affine(xs, ys):
    A = np.vstack([xs, np.ones_like(xs)]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    return _nrms(ys, a * xs + b), a, b


def monotone(ys):
    d = np.diff(ys)
    return bool(np.all(d >= -1e-7 * max(1.0, float(np.max(np.abs(ys)))))) and float(ys[-1] - ys[0]) > 1e-9


def fit(xs, ys):
    """Classify y = f(x), sampled at xs (positive, increasing in the probe).
    Returns a Fit or None (opaque)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys) & np.isfinite(xs)
    if ok.sum() < 8:
        return None
    xs, ys = xs[ok], ys[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if float(xs[-1] - xs[0]) <= 1e-9 or not monotone(ys):
        return None
    # normalise the input to [0, 1] when it is a positive range; scale is free
    lo, hi = float(xs[0]), float(xs[-1])
    if lo < -1e-9:
        return None
    xn = xs / hi
    yn = ys / max(abs(float(ys[-1])), 1e-12)
    per_class = {}
    ea, a, b = _affine(xn, yn)
    per_class["affine"] = (ea, Fit("affine (scale and offset)", "affine", (), ea))
    for name, cls, el, fn, grid in LIBRARY:
        e, s = _fit_one(xn, yn, fn, grid)
        if e < per_class.get(cls, (float("inf"),))[0]:
            per_class[cls] = (e, Fit(name, cls, el, e, s / hi if hi else s))
    ranked = sorted(per_class.values(), key=lambda t: t[0])
    best_e, best = ranked[0]
    # identity-like classes (affine and tonemaps) agree on the group element
    def element(f):
        return f.element
    rivals = [(e, f) for e, f in ranked[1:] if element(f) != element(best)]
    if best_e > TOL:
        return None
    if rivals and rivals[0][0] < max(MARGIN * best_e, 2 * TOL):
        return None
    best.runner_up = rivals[0][1].name if rivals else ""
    return best


def fit_affine(xs, ys, tol=1e-4):
    """Exact affine fit for range remaps: returns (a, b) or None."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3 or float(np.ptp(xs)) < 1e-9:
        return None
    e, a, b = _affine(xs, ys)
    if not math.isfinite(e) or e > tol:
        if float(np.ptp(ys)) < 1e-12:
            return None
        return None
    return float(a), float(b)
