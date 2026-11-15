"""Watch an AVG checkpoint live (window) or record it to MP4. Prints why each episode ended.

  python watch_avg.py --ckpt avg_v2_s0                       # latest checkpoint in folder, live window
  python watch_avg.py --ckpt avg_v2_s0/<file>.pt --slow 3    # 3x slow motion
  python watch_avg.py --ckpt avg_v2_s0 --video s0.mp4        # record instead (headless: MUJOCO_GL=egl)
Needs inspect_checkpoints.py, humanoid_env.py and avg.py in the same folder.
"""
import argparse, glob, os, time
import numpy as np
from inspect_checkpoints import load_agent, make_policy
from humanoid_env import make_robot_env

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="a .pt file, or a results folder (uses the newest checkpoint)")
ap.add_argument("--xml", default="humanoid2.xml"); ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor for the live window")
ap.add_argument("--stochastic", action="store_true", help="sample actions like training does")
ap.add_argument("--video", default=None, help="write an MP4 here instead of opening a window")
ap.add_argument("--distance", type=float, default=1.5); ap.add_argument("--azimuth", type=float, default=90)
ap.add_argument("--elevation", type=float, default=-15)
a = ap.parse_args()

path = a.ckpt
if os.path.isdir(path):
    path = sorted(glob.glob(os.path.join(path, "*_model*.pt")), key=os.path.getmtime)[-1]
cam = dict(trackbodyid=1, distance=a.distance, azimuth=a.azimuth, elevation=a.elevation)
env = make_robot_env(a.xml, render_mode="rgb_array" if a.video else "human", default_camera_config=cam)
agent, step, mean, var = load_agent(path, env)
act = make_policy(agent, mean, var, env, a.stochastic)
dt, frames = env.unwrapped.dt, []
print(f"checkpoint: {path}  (training step {step})  mode: {'stochastic' if a.stochastic else 'deterministic'}")

for ep in range(a.episodes):
    o, _ = env.reset(seed=2000 + ep); t, R = 0, 0.0
    while True:
        t0 = time.time()
        o, r, te, tr, info = env.step(act(o)); R += r; t += 1
        frame = env.render()
        if a.video: frames.append(frame)
        else:
            lag = dt * a.slow - (time.time() - t0)
            if lag > 0: time.sleep(lag)
        d = env.unwrapped.data
        if t % 50 == 0:
            print(f"  ep{ep+1} t={t:4d}  vx={info.get('x_velocity', float(d.qvel[0])):+.2f}  z={d.qpos[2]:.3f}  up={d.xmat[1][8]:.2f}")
        if te or tr: break
    z, up = float(d.qpos[2]), float(d.xmat[1][8])
    if tr: why = "survived the full 1000 steps"
    elif z < 0.33: why = f"height floor (z={z:.3f})"
    elif up < 0.85: why = f"tilt (up={up:.2f}, pelvis ~{np.degrees(np.arccos(np.clip(up, -1, 1))):.0f} deg off vertical)"
    else: why = f"other (z={z:.3f})"
    print(f"ep{ep+1}: {t} steps, return {R:.0f}, ended by: {why}")

if a.video:
    import imageio
    imageio.mimsave(a.video, frames, fps=round(1 / dt)); print(f"saved {a.video}")
env.close()
