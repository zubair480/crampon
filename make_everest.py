"""Generate the Everest Base Camp approach as a MuJoCo heightfield.

Built from what the route actually is, not a generic peak. On the ground the
trek follows the LATERAL MORAINE of the Khumbu Glacier -- a ridge of rock
debris beside the ice -- and the lower glacier is rock- and dust-covered,
described as looking more like a grey desert than a river of ice. Ice seracs
flank the moraine near Base Camp, and the Everest-Nuptse-Lhotse walls rise
behind. Sources: Khumbu Glacier trek guides, see README.

Layout, 400 m square, looking north toward the massif:

    +y (far)   Everest pyramid + Nuptse wall  -- the backdrop, unwalkable
      |        Khumbu icefall apron spilling toward the valley
      |        glacier floor (rubble-covered, gently undulating)
      |        LATERAL MORAINE crest  <- the trail; the robot walks this
    -y (near)  outwash slope

The moraine crest is graded to a walkable angle deliberately. Everything else
is at real Himalayan angles, which is to say unwalkable -- that contrast is
the point.
"""

import argparse
import struct

import numpy as np


def _smooth(a, k=3):
  """Cheap box blur, repeated -- avoids a scipy dependency."""
  out = a.astype(float)
  for _ in range(k):
    p = np.pad(out, 1, mode="edge")
    out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
           + 4 * p[1:-1, 1:-1]) / 8.0
  return out


def fbm(shape, octaves=6, seed=0):
  rng = np.random.default_rng(seed)
  out = np.zeros(shape)
  amp, total = 1.0, 0.0
  for o in range(octaves):
    n = max(2, int(3 * 2 ** o))
    coarse = rng.normal(size=(n, n))
    yi = np.linspace(0, n - 1, shape[0])
    xi = np.linspace(0, n - 1, shape[1])
    y0 = np.clip(np.floor(yi).astype(int), 0, n - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, n - 2)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    layer = (coarse[np.ix_(y0, x0)] * (1 - fx) * (1 - fy)
             + coarse[np.ix_(y0, x0 + 1)] * fx * (1 - fy)
             + coarse[np.ix_(y0 + 1, x0)] * (1 - fx) * fy
             + coarse[np.ix_(y0 + 1, x0 + 1)] * fx * fy)
    out += amp * layer
    total += amp
    amp *= 0.5
  return out / total


def build(res=768, size_m=400.0, relief_m=260.0, seed=11,
          moraine_y_m=-70.0, moraine_w_m=14.0):
  """Return (heights01, meta). Heights are normalised to [0,1]."""
  mpp = size_m / res
  yy, xx = np.mgrid[0:res, 0:res]
  X = (xx - (res - 1) / 2.0) * mpp          # metres, +x east
  Y = (yy - (res - 1) / 2.0) * mpp          # metres, +y north (into the massif)

  # --- valley floor: gently undulating rubble ---
  floor = 6.0 + 4.0 * fbm((res, res), 5, seed) + 1.2 * fbm((res, res), 7, seed + 3)

  # --- Everest: a pyramid at the far side, steep and dominant ---
  ev_c = np.array([30.0, 235.0])
  d_ev = np.hypot(X - ev_c[0], Y - ev_c[1])
  everest = np.clip(1.0 - d_ev / 170.0, 0, 1) ** 1.35 * relief_m
  # Ridge texture so the faces are not smooth planes.
  everest *= 0.86 + 0.14 * (1.0 - np.abs(fbm((res, res), 6, seed + 1)))

  # --- Nuptse: a long wall running east-west, closer than Everest ---
  nuptse = np.clip(1.0 - np.abs(Y - 150.0) / 60.0, 0, 1) ** 1.5
  nuptse *= np.clip(1.0 - np.abs(X + 90.0) / 190.0, 0, 1) ** 0.6
  nuptse *= relief_m * 0.62
  nuptse *= 0.88 + 0.12 * (1.0 - np.abs(fbm((res, res), 6, seed + 2)))

  # --- icefall apron: the ice spilling out of the cwm toward the valley ---
  apron = np.clip(1.0 - np.abs(Y - 95.0) / 55.0, 0, 1) ** 1.2
  apron *= np.clip(1.0 - np.abs(X - 20.0) / 120.0, 0, 1)
  apron *= 46.0
  # Chaotic blocky texture: the icefall is broken, not smooth.
  apron *= 0.75 + 0.5 * np.abs(fbm((res, res), 7, seed + 4))

  h = floor + np.maximum(everest, nuptse) + apron

  # --- lateral moraine: a raised ridge of debris the trail follows ---
  # Sinuous, so it reads as a real landform rather than a straight embankment.
  wander = 16.0 * np.sin(X / 95.0) + 7.0 * np.sin(X / 41.0 + 1.3)
  dist = np.abs(Y - (moraine_y_m + wander))
  crest = np.exp(-0.5 * (dist / (moraine_w_m * 0.75)) ** 2)
  ridge_h = 11.0 + 2.2 * np.sin(X / 26.0)
  h = h * (1 - crest) + (floor + ridge_h) * crest
  # Rough rubble on the flanks, smoothed on the crest itself so it is walkable.
  rubble = 0.55 * fbm((res, res), 8, seed + 5)
  h += rubble * (1 - crest * 0.92)

  # Grade the crest: this is the only walkable line in the scene.
  band = np.exp(-0.5 * (dist / (moraine_w_m * 0.42)) ** 2)
  h = h * (1 - band) + _smooth(h, 7) * band

  h -= h.min()
  hmax = h.max()
  h /= hmax
  meta = dict(res=res, size_m=size_m, relief_m=float(hmax), mpp=mpp,
              moraine_y_m=moraine_y_m, moraine_w_m=moraine_w_m,
              wander=lambda x: 16.0 * np.sin(x / 95.0) + 7.0 * np.sin(x / 41.0 + 1.3))
  return h.astype(np.float32), meta


def write_hfield(path, h):
  with open(path, "wb") as f:
    f.write(struct.pack("<ii", h.shape[0], h.shape[1]))
    f.write(h.astype("<f4").tobytes())


def report(h, meta):
  mpp, relief = meta["mpp"], meta["relief_m"]
  gy, gx = np.gradient(h * relief, mpp)
  slope = np.degrees(np.arctan(np.hypot(gy, gx)))
  res = meta["res"]
  yy, xx = np.mgrid[0:res, 0:res]
  X = (xx - (res - 1) / 2.0) * mpp
  Y = (yy - (res - 1) / 2.0) * mpp
  d = np.abs(Y - (meta["moraine_y_m"] + meta["wander"](X)))
  trail = d < meta["moraine_w_m"] * 0.35
  print(f"  relief          : {relief:.0f} m over {meta['size_m']:.0f} m")
  print(f"  whole scene     : mean slope {slope.mean():5.1f} deg  p90 "
        f"{np.percentile(slope, 90):5.1f}")
  print(f"  moraine trail   : mean slope {slope[trail].mean():5.1f} deg  p90 "
        f"{np.percentile(slope[trail], 90):5.1f}   <- the walkable line")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--res", type=int, default=768)
  ap.add_argument("--size-m", type=float, default=400.0)
  ap.add_argument("--relief-m", type=float, default=260.0)
  ap.add_argument("--seed", type=int, default=11)
  ap.add_argument("--out", default="hfield_everest.bin")
  a = ap.parse_args()
  h, meta = build(a.res, a.size_m, a.relief_m, a.seed)
  write_hfield(a.out, h)
  print(f"wrote {a.out}  {h.shape[0]}x{h.shape[1]}  ({meta['mpp']:.2f} m/cell)")
  report(h, meta)
