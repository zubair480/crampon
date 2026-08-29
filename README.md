# crampon

**Unitree G1 humanoid recovering from falls on ice.** RL policy trained under
Himalaya conditions: friction mu 0.02-0.35, -20C actuator derating, high-wind
gusts.

Himalaya Robotics Hack, Aug 29-30 2026 - **Track 1: Movement**.

## Why getting up is the problem worth solving

A humanoid that falls on ice at altitude and cannot stand back up is a dead
mission. Stock getup behaviour assumes a high-friction floor: the limbs push
sideways, the contact force leaves the friction cone, and the robot skates
instead of rising. We train a policy that keeps ground reaction forces inside a
narrow cone, and that infers how slick the surface is from proprioception alone.

MuJoCo Playground ships **no G1 getup task** - only walking
(`G1JoystickFlatTerrain`). Getup exists solely for quadrupeds (`Go1Getup`,
`SpotGetup`). The task is ported from the Go1 template onto the G1.

## Extreme-condition model

`crampon/randomize_ice.py` replaces Playground's stock G1 randomizer, which
samples floor friction from `U(0.4, 1.0)` - a dry gym floor.

| effect | how it is modelled |
| --- | --- |
| ice / packed snow | `pair_friction` sampled over `U(0.02, 0.35)` |
| cold: thickened grease | `dof_frictionloss` x`U(1.5, 3.0)`, `dof_damping` x`U(1.2, 2.0)` |
| cold: battery sag | servo gain `kp` x`U(0.7, 0.9)` |
| wind | `xfrc_applied` gusts on the torso (in `step()`, not here) |

**Gotcha worth knowing:** the G1's actuators are *unlimited-force position
servos* - `actuator_forcerange` is all zeros and `actuator_forcelimited` is
False. Scaling force range to model a cold torque limit is a silent no-op that
yields NaN. The knob that actually bites is `gainprm[:,0]`, mirrored as `-kp`
in `biasprm[:,1]`; both must be scaled together or the servo goes unstable.

```python
from crampon.randomize_ice import make_randomizer, ICE, MIXED, DRY

randomize = make_randomizer(MIXED)        # training distribution
randomize = make_randomizer((0.05, 0.05)) # pin mu exactly, for eval sweeps
```

The same factory drives training and the success-rate-vs-mu evaluation curve.

## Setup

Requires Python 3.10+ (3.12 recommended); CUDA 12 for GPU.

```bash
git clone https://github.com/zubair480/crampon.git && cd crampon
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

### Read this before you debug for an hour

A plain `pip install playground` today resolves `jax` to **0.11.1**, which
removed `jax.device_put_replicated`. Brax 0.14.2 still calls it, so every
training run dies at `brax/training/agents/ppo/train.py:756` - *after* printing
the full config block, and the console script swallows the traceback, so you
get a bare exit 1 with no error. `requirements.txt` pins `jax<0.10` (0.9.2 is
the newest that works). 0.10.2 is also broken.

Verify:

```bash
python -c "import jax; print(jax.default_backend())"          # -> gpu
python -c "import jax; print(hasattr(jax,'device_put_replicated'))"  # -> True
```

## Training

No GPU locally; all real training runs on Hugging Face Jobs.

```bash
hf jobs uv run --flavor l4x1     --timeout 20m jobs/hf_smoke.py   # validate stack first
hf jobs uv run --flavor a100-large --timeout 4h  jobs/train_getup.py
```

`jobs/hf_smoke.py` checks the GPU is visible, the jax pin still satisfies brax,
the G1 loads, the randomizer produces sane batched models, and PPO steps -
before any budget goes into a long run.

## Credits

Built on [MuJoCo Playground](https://playground.mujoco.org/) and
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0). The getup task is ported from Playground's `Go1Getup`.
