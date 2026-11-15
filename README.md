# Humanoid-avg
training my custom humanoid with the AVG method

# Incremental policy gradients (AVG) on a custom bipedal humanoid

This project trains a custom bipedal humanoid (`humanoid2.xml`, MuJoCo) to walk with **Action-Value Gradient (AVG)**, an incremental deep policy-gradient method with batch size 1, no replay buffer and no target networks (Vasan et al., 2024). It builds on the authors' reference implementation and compares the behaviour with a batched PPO baseline I trained earlier on the same robot ([Humanoid-RL](https://github.com/Iqmaa/Humanoid-RL)).

**Short version.** With the original Humanoid reward, AVG learned to stand still. With a revised reward, three seeds learned forward locomotion, but training was unstable: returns collapsed abruptly and then recovered, repeatedly. Each seed has intermediate checkpoints that survive most of a 1,000-step episode while moving forward (seed 0 at 4.75M steps, seed 1 at 4.25M, seed 2 at 2.25M), but no run is stable. Seed 2 collapsed at 2.75M steps and never regained long survival: it ended with a fast one-legged gait (observed in video) that falls after roughly 250 steps. On 50 held-out episodes, the selected checkpoints meet a walking criterion (1,000 steps, at least 0.8 m/s) in 86% (seed 0), 100% (seed 1) and 84% (seed 2) of episodes. The core AVG update was not modified. What I changed is the environment, the reward, checkpointing, evaluation and tooling.

<!-- Videos: record with record_videos.py and add the links here -->
| Seed | Checkpoint | Clip |
|---|---|---|
| 0 | 4.75M steps | `videos/avg_v2_s0_step4750000.mp4` |
| 1 | 4.25M steps | `videos/avg_v2_s1_step4250000.mp4` |
| 2 | 2.25M steps (best checkpoint) | `videos/avg_v2_s2_step2250000.mp4` |
| 2 | 5.0M steps (after its collapse; one-legged gait) | `videos/avg_v2_s2_step5000000.mp4` |

`videos/manifest.csv` lists the length, mean speed and end reason of every recorded episode.

## Robot and task

- Custom ~0.39 m biped: per leg hip roll, hip pitch, knee and ankle pitch (no ankle roll); waist pitch, neck yaw, and arms. Actuators are limited to ±0.4.
- Gymnasium `Humanoid-v5` with the custom XML, a compact 43-number observation (`qpos[2:]` + `qvel`), episodes capped at 1,000 steps.
- Wrappers: terminate when the pelvis tilts more than about 32° from vertical, and a heading penalty (`2·sin²(yaw)`) against turning away from +x.
- Reward, revised (used for all reported runs): alive bonus 1.0, plus `4.0·clip(vx, −0.5, 1.0)`, minus Humanoid's control cost. The original used the Humanoid-v5 defaults: alive bonus 5.0 plus `1.25·vx`.

## What happened

| Phase | Observation | Cause | Fix |
|---|---|---|---|
| First 6M-step runs (original reward) | AVG stood still, earning about 4.85 reward per step with forward speed near 0 | The 5.0 alive bonus dwarfs `1.25·vx` (1.0 at 0.8 m/s), so surviving is worth more than walking | Alive bonus 1.0 and a capped velocity reward |
| Same runs, evaluation | Seed 2 trained to 1,000-step episodes yet evaluated at 14 steps | The evaluation script skipped the actor's penultimate-layer L2 normalization and did not clip actions as in training | Evaluation mirrors `Actor.forward`; fixed evaluation then gave 1,000 steps / 4,873 return, matching training |
| Revised runs | Hopping, then falling, in the first 1–2M steps; walking-like gaits later | Early-training behaviour under a dense velocity reward | More training; an anti-hopping reward variant (`ROBOT_REWARD=gait`) is prepared but was not used for the results here |
| Revised runs | Abrupt collapses of episode length, then recovery | Not yet identified (see below) | Per-checkpoint inspection and selection (`inspect_checkpoints.py`) |

![Training curves](figures/training_curves.png)

*Top row: original reward (smooth rise to a stationary equilibrium). Bottom row: revised reward (rolling mean over 200 training episodes; the vertical drops are collapses).*

### Instability

Under the revised reward, rolling episode length drops from several hundred to near zero within a few thousand steps. By a simple threshold (rolling length falling below 100 after exceeding 300), seed 0 had 2 such events, seed 1 had 7 (each recovering in 50–80K steps), and seed 2 had 1 (about 1.7M steps to recover). Evaluating consecutive 250K-step checkpoints shows the same pattern: deterministic episode length swings between 7 and 1,000.

![Seed 0 checkpoints](figures/summary_seed0.png)
![Seed 1 checkpoints](figures/summary_seed1.png)
![Seed 2 checkpoints](figures/summary_seed2.png)

| Seed | Best checkpoint | Det. length / speed | Stoch. length / speed | Latest evaluated (5.0M) |
|---|---|---|---|---|
| 0 | 4.75M | 947 / 1.59 m/s | 938 / 1.63 m/s | 546 / 1.23 m/s |
| 1 | 4.25M | 1000 / 1.16 m/s | 935 / 1.13 m/s | 927 / 0.95 m/s |
| 2 | 2.25M | 888 / 1.61 m/s | 988 / 1.62 m/s | 266 / 1.79 m/s |

Each row is the mean of 20 episodes at fixed reset seeds, and the same episodes were used to pick the best checkpoint, so these numbers could be optimistic (see the held-out check below). Seed 2 after its collapse (3.0M to 5.0M) moves fast (mean 1.4 m/s) but survives only 55 to 277 steps, so the one-legged gait is not a stable walk.

### Held-out check

| Seed | Checkpoint | Episodes surviving 1,000 steps | Mean length ± SD | Mean speed | Walking criterion met |
|---|---|---|---|---|---|
| 0 | 4.75M | 43 / 50 | 929 ± 216 | 1.58 m/s | 86% |
| 1 | 4.25M | 50 / 50 | 1000 ± 0 | 1.15 m/s | 100% |
| 2 | 2.25M | 42 / 50 | 940 ± 170 | 1.62 m/s | 84% |

Deterministic policy, 50 fresh episodes (reset seeds 5000 to 5049) that played no part in choosing the checkpoints. The walking criterion is: 1,000 steps survived, mean forward speed of at least 0.8 m/s, mean root height of at least 0.35 m. Every full-length episode also met the speed criterion. Return was about 4.6 per step for all three. The held-out numbers match or exceed the selection numbers, so selecting the best of 20 checkpoints did not noticeably inflate these three. The episodes differ only in initial-state noise, so this says nothing about robustness to pushes or terrain, and it tests fixed snapshots, not the stability of training. Speeds above 1.0 m/s are not penalized, only not rewarded further.

## Changes relative to the reference AVG implementation

A line-by-line diff shows the actor, critic and `update()` (the AVG algorithm) and all default hyperparameters unchanged (actor lr 0.0063, critic lr 0.0087, γ 0.99, α 0.07, Adam β₁ 0). The changes are:

| Change | Why |
|---|---|
| `make_robot_env` hook and `--xml` | Train on the custom robot with shared wrappers |
| Checkpoints now store step, observation-normalization statistics, TD-error scaler and RNG state | The reference `save()` stores only network and optimizer weights, so a policy could not be replayed or evaluated faithfully |
| `--resume` | Continue a run from a checkpoint |
| Checkpoint filenames zero-padded, saved at `(t+1) % N == 0` | The reference names (`human_format_numbers`) round to the nearest K/M, so e.g. 1.25M and 1.5M would both be `1M`; it also saved at step 0 |
| Logged step counter `T` is `t+1` | Consistent with checkpoint steps |

## AVG versus PPO

| | AVG (this work) | PPO (SB3, original project) |
|---|---|---|
| Update | One transition per update | 8 envs × 512 steps per rollout, 5 epochs |
| Stored data | None | Rollout buffer |
| Normalization | Running observation normalization; TD-error scaling | `VecNormalize` on observations and rewards |
| Discount | 0.99 | 0.95 |
| Parallelism | 1 environment, 1 thread | 8 processes |
| Wall-clock | ≈134 steps/s (12.4 h per 6M steps) | ≈3 h per 6M steps (hardware may differ) |
| Hyperparameters | Reference defaults, untouched | Custom PPO baseline |

Both algorithms exploited the original reward, in opposite ways: AVG stood still, and PPO hopped forward at about 2 m/s and fell after roughly 260 steps (from my earlier PPO evaluation). This suggests reward design mattered more than the algorithm here. PPO was **not** re-run on the revised reward, so there is no like-for-like performance comparison yet. The AVG paper reports smooth, seed-aggregated Humanoid-v4 curves reaching roughly 4,000–4,500 return over 10M steps; the per-seed collapses I see are not visible in aggregate curves, but my robot, reward and seed count differ, so I treat this as a question rather than a contradiction.

## Limitations

Three seeds; the held-out check varies only initial-state noise, and its walking criterion is based on length, speed and height, not a measure of gait quality; the revised-reward runs stopped near 5M steps rather than 6M; no ablation isolates which reward change mattered; the cause of the collapses is untested.

## Next steps

Log TD-error scale, Q magnitude and policy entropy around collapses; test lower or decayed step sizes and frozen observation statistics; add a foot-alternation reward to address the one-legged gait; rerun PPO on the revised reward for a matched comparison; more seeds.

## Usage

```bash
python avg.py --env MyHumanoid --xml humanoid2.xml --seed 0 --N 6000000 --checkpoint 250000 --save_model --results_dir avg_v2_s0 > avg_v2_s0.log
python inspect_checkpoints.py --dir avg_v2_s0 --log avg_v2_s0.log --episodes 20   # evaluate every checkpoint, flag failure modes
python watch_avg.py --ckpt avg_v2_s0 --slow 3                                     # live viewer
python record_videos.py --runs avg_v2_s0:4750000 avg_v2_s1:4250000                # MP4s
python avg.py ... --resume avg_v2_s0/<checkpoint>.pt                              # continue training
```

## Repository contents

`avg.py` (modified reference AVG), `humanoid_env.py` (task, reward, wrappers), `humanoid2.xml` + `meshes/`, `evaluate_policy.py`, `inspect_checkpoints.py`, `watch_avg.py`, `record_videos.py`, `smoke_test_env.py`, `smoke_test_gait.py`, `figures/`.

## Reference

Vasan, G., Elsayed, M., Azimi, A., He, J., Shariar, F., Bellinger, C., White, M., Mahmood, A. R. *Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers.* 2024.
