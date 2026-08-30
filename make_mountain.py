"""Generate a complete ice mountain as a MuJoCo heightfield.

The supplied Khumbu DEM is a 120 m patch of one slope, so any wide camera shot
shows its edges and it reads as a tile rather than a mountain. This builds a
whole peak: a summit, radial ridgelines, and -- critically -- a graded traverse
bench the robot can actually hike, because a bare cone at Himalayan angles is
unwalkable everywhere.

Output is MuJoCo's hfield binary format: int32 nrow, int32 ncol, then
nrow*ncol float32 in [0,1].

    python make_mountain.py --res 512 --size-m 400 --peak-m 120
"""

import argparse
import struct

import numpy as np


def fbm(shape, octaves=6, seed=0):
  """Fractal noise, built by summing smoothed random fields at halving scale."""
  rng = np.random.default_rng(seed)
  out = np.zeros(shape)
  amp, total = 1.0, 0.0
  for o in range(octaves):
    n = max(2, int(4 * 2 ** o))
    coarse = rng.normal(size=(n, n))
    # Bilinear upsample to full resolution.
    yi = np.linspace(0, n - 1, shape[0])
    xi = np.linspace(0, n - 1, shape[1])
    y0 = np.clip(np.floor(yi).astype(int), 0, n - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, n - 2)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    a = coarse[np.ix_(y0, x0)]
    b = coarse[np.ix_(y0, x0 + 1)]
    c = coarse[np.ix_(y0 + 1, x0)]
    d = coarse[np.ix_(y0 + 1, x0 + 1)]
    layer = (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
             + c * (1 - fx) * fy + d * fx * fy)
    out += amp * layer
    total += amp
    amp *= 0.5
  return out / total


def build(res=512, size_m=400.0, peak_m=120.0, seed=7,
          bench_r_m=110.0, bench_w_m=26.0, bench_grade=0.10):
  """Return (heights01, meta). bench_* carve a walkable traverse."""
  y, x = np.mgrid[0:res, 0:res]
  cx = cy = (res - 1) / 2.0
  mpp = size_m / res
  rx = (x - cx) * mpp
  ry = (y - cy) * mpp
  r = np.hypot(rx, ry)
  theta = np.arctan2(ry, rx)

  # Cone: full height at the centre, zero at the rim. Squared falloff gives a
  # broad base and a steeper summit, which is what a real peak looks like.
  rmax = size_m / 2.0
  cone = np.clip(1.0 - r / rmax, 0.0, 1.0) ** 1.6

  # Ridgelines: fold noise so crests are sharp rather than rounded.
  ridge = 1.0 - np.abs(fbm((res, res), octaves=6, seed=seed))
  ridge = ridge ** 2
  # Six radial spurs, fading out near the summit.
  spurs = 0.5 + 0.5 * np.cos(6 * theta)
  spurs *= np.clip((r - 0.15 * rmax) / (0.5 * rmax), 0, 1)

  h = cone * (0.80 + 0.14 * ridge + 0.10 * spurs)
  h += 0.05 * fbm((res, res), octaves=7, seed=seed + 1) * cone

  # Graded traverse: a level-ish bench circling the peak. Without it the
  # mountain is unwalkable everywhere and the robot simply slides off.
  band = np.exp(-0.5 * ((r - bench_r_m) / (bench_w_m / 2.0)) ** 2)
  bench_h = np.clip(1.0 - bench_r_m / rmax, 0.0, 1.0) ** 1.6 * 0.80
  # Slight downhill along the traverse so the hike has a direction.
  bench_h = bench_h + bench_grade * (theta / np.pi) * (bench_w_m / size_m)
  h = h * (1 - band) + bench_h * band

  h -= h.min()
  h /= h.max()
  meta = dict(res=res, size_m=size_m, peak_m=peak_m, mpp=mpp,
              bench_r_m=bench_r_m)
  return h.astype(np.float32), meta


def write_hfield(path, h):
  with open(path, "wb") as f:
    f.write(struct.pack("<ii", h.shape[0], h.shape[1]))
    f.write(h.astype("<f4").tobytes())


def report(h, meta):
  mpp, peak = meta["mpp"], meta["peak_m"]
  gy, gx = np.gradient(h * peak, mpp)
  slope = np.degrees(np.arctan(np.hypot(gy, gx)))
  res = meta["res"]
  yy, xx = np.mgrid[0:res, 0:res]
  c = (res - 1) / 2.0
  r = np.hypot((xx - c) * mpp, (yy - c) * mpp)
  band = np.abs(r - meta["bench_r_m"]) < 8.0
  print(f"  overall slope: mean {slope.mean():5.1f} deg  p90 {np.percentile(slope,90):5.1f}")
  print(f"  traverse bench: mean {slope[band].mean():5.1f} deg  p90 "
        f"{np.percentile(slope[band],90):5.1f}   <- where the robot hikes")
  print(f"  summit height : {h.max()*peak:.1f} m over {meta['size_m']:.0f} m")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=512)
  ap.add_argument("--size-m", type=float, default=400.0)
  ap.add_argument("--peak-m", type=float, default=120.0)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--bench-r", type=float, default=110.0)
  ap.add_argument("--out", default="hfield_mountain.bin")
  a = ap.parse_args()
  h, meta = build(a.res, a.size_m, a.peak_m, a.seed, bench_r_m=a.bench_r)
  write_hfield(a.out, h)
  print(f"wrote {a.out}  {h.shape[0]}x{h.shape[1]}  ({meta['mpp']:.2f} m/cell)")
  report(h, meta)
