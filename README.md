# crampon

**Keeping a Unitree G1 on its feet on ice.** An RL locomotion policy trained
under Himalaya conditions — friction down to μ=0.01, −20 °C actuator
derating, and gusts modelled from the drag equation at 8000 m air density.

Himalaya Robotics Hack, 29–30 Aug 2026 — **Track 1: Movement**.

![survival vs friction](sweep.png)

## Result

Both policies evaluated under **identical** conditions — same wind, same cold,
same 24 seeds. Survival is steps completed out of 500 before the torso tips
past horizontal.

| μ | baseline | ours | delta | ours ahead on |
|---|---|---|---|---|
| 0.010 | 220.5 | **432.5** | +96% | 21/24 seeds |
| 0.015 | 154.2 | **333.5** | +116% | 21/24 |
| 0.020 | 122.5 | **258.0** | +111% | 19/24 |
| 0.030 | 66.6 | **212.2** | +219% | 23/24 |
| 0.050 | 48.9 | **154.3** | +216% | 22/24 |
| 0.100 | 49.0 | **325.3** | +564% | 24/24 |
| 0.350 | 500.0 | 500.0 | — | both saturate |

**2× to 6.6× longer on ice, with no regression on normal ground.** The policy
did not trade away ordinary walking to buy ice performance.

## Why this matters on a mountain

A humanoid that falls at 8000 m and cannot recover is not a broken robot, it
is a body someone has to climb up and retrieve. Every metre of extra footing
is a rescue not mounted.

## What we actually modelled

Stock Playground samples floor friction from `U(0.4, 1.0)` — a dry gym floor.
Everything below replaces that with the mountain.

| effect | how it is modelled |
| --- | --- |
| ice / packed snow | contact-pair friction over `U(0.02, 0.35)` |
| cold: thickened grease | joint dry friction ×`U(1.5, 3.0)`, damping ×`U(1.2, 2.0)` |
| cold: battery sag | servo gain `kp` ×`U(0.7, 0.9)` |
| wind | sustained force on the torso, `F = ½ρCdAv²` |

Wind is a **force**, not the impulse Playground's `push_config` applies —
surviving a kick is a different skill from leaning into a sustained headwind.
ρ = 0.45 kg/m³ is air density at ~8000 m, about a third of sea level, so a
200 km/h gale up there pushes with 348 N against a robot weighing 327 N: a
sideways shove slightly harder than gravity. Gusts follow an
Ornstein–Uhlenbeck process so they stay correlated over ~2 s rather than being
white noise a policy can average away.

## The finding

**Naive domain randomization fails.** Training from scratch on the full ice
band diverges: an exploring policy on near-frictionless ground reaches
enormous velocities, enormous observations, and the network goes NaN. Our
first attempt returned `reward=nan` at every evaluation across 85M steps.

What works is a **curriculum**: learn to walk on normal ground first, then
warm-start onto ice with the friction band annealed and observations clipped.
Full 200 km/h wind stays an *evaluation* condition, never a training one.

## Deployment

The policy runs on **native MuJoCo at 462 Hz on a laptop CPU with no GPU** —
2.17 ms per control step against a 20 ms budget, measured. MJX is for training
thousands of robots on a GPU; deployment is single-robot and native, which is
what Jetson Thor would run. Same code path for the live demo and the robot.

## Run it

```bash
git clone https://github.com/zubair480/crampon.git && cd crampon
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python play.py --filename policy-ice-s1.pkl --mu 0.05 --wind 12
```

Live interactive viewer: W/S walk, A/D strafe, Q/E turn, X stop, R reset,
ctrl+drag to shove the robot and try to knock it over.

```bash
python fair_eval.py       # reproduce the table above
python make_plot.py       # reproduce the figure
python render_compare.py  # reproduce the side-by-side video
```

### Read this before you debug for an hour

A plain `pip install playground` resolves `jax` to 0.11, which removed
`jax.device_put_replicated`. Brax 0.14.2 still calls it, so every training run
dies at `ppo/train.py:756` — *after* printing the full config, with the
traceback swallowed, leaving a bare exit 1. `requirements.txt` pins
`jax<0.10`; 0.9.2 is the newest that works.

## Also in here

`crampon/getup_model.py` gives the G1 full-body floor contacts. Playground
declares five contact pairs and only two reach the ground, one per foot —
correct for walking, fatal for fall recovery, since a fallen humanoid passes
straight through the ice. The trap is that this model resolves contact through
explicit `<pair>` elements rather than `contype`/`conaffinity`, so adding
collision geoms alone does nothing at all. Drop test, same fall:

```
feetonly (stock)   torso z = -0.749 m   sank through the floor
getup (patched)    torso z = +0.098 m   lying on it
```

`crampon/getup_env.py` ports Playground's `Go1Getup` onto the humanoid —
Playground ships getup for quadrupeds only. Training was still in progress at
submission.

## Credits

Built on [MuJoCo Playground](https://playground.mujoco.org/) and
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache-2.0. Trained on Hugging Face Jobs. The getup task is ported from
Playground's `Go1Getup`.
