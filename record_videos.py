"""Record MP4s of chosen AVG checkpoints. Camera tracks the robot's root body (trackbodyid=1).

  # one clip per run:step pair  (folder name : training step)
  python record_videos.py --runs avg_v2_s0:4750000 avg_v2_s1:4250000 avg_v2_s2:3000000

  # several checkpoints of one seed, 3 episodes each, 3x slow motion, bigger frames
  python record_videos.py --runs avg_v2_s0:1000000 avg_v2_s0:3750000 avg_v2_s0:4750000 --episodes 3 --slow 3 --width 960 --height 720

  # headless machine:  MUJOCO_GL=egl python record_videos.py ...
Needs: pip install imageio imageio-ffmpeg pillow
Writes videos/<run>_step<N>.mp4 and videos/manifest.csv (length, speed and end reason of EVERY recorded episode,
so clips can be reported honestly rather than cherry-picked).
"""
import argparse, csv, glob, os
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from inspect_checkpoints import load_agent, make_policy
from humanoid_env import make_robot_env

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="folder:step pairs, e.g. avg_v2_s1:4250000")
ap.add_argument("--xml", default="humanoid2.xml"); ap.add_argument("--out", default="videos")
ap.add_argument("--episodes", type=int, default=1, help="episodes per clip, played back to back")
ap.add_argument("--seed0", type=int, default=2000, help="first episode reset seed (then +1 each)")
ap.add_argument("--stochastic", action="store_true", help="sample actions like training (default: deterministic mean)")
ap.add_argument("--slow", type=float, default=1.0); ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480); ap.add_argument("--distance", type=float, default=1.5)
ap.add_argument("--azimuth", type=float, default=90); ap.add_argument("--elevation", type=float, default=-15)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cam = dict(trackbodyid=1, distance=a.distance, azimuth=a.azimuth, elevation=a.elevation)
env = make_robot_env(a.xml, render_mode="rgb_array", width=a.width, height=a.height, default_camera_config=cam)
try: font = ImageFont.load_default(size=max(14, a.height // 28))
except TypeError: font = ImageFont.load_default()
dt = env.unwrapped.dt; fps = max(1, round(1 / dt / a.slow))

def caption(frame, lines):
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        y = 6 + i * (font.size + 4 if hasattr(font, "size") else 14)
        d.text((8, y), t, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)

manifest = []
for spec in a.runs:
    folder, step = spec.rsplit(":", 1); step = int(step)
    hits = glob.glob(os.path.join(folder, f"*_model_{step:08d}.pt"))
    if not hits:
        have = sorted(int(os.path.basename(p).rsplit("_", 1)[1][:8]) for p in glob.glob(os.path.join(folder, "*_model_*.pt")))
        print(f"!! no checkpoint at step {step} in {folder}. Available: {have}"); continue
    agent, ck_step, mean, var = load_agent(hits[0], env)
    act = make_policy(agent, mean, var, env, a.stochastic)
    frames, label = [], f"{os.path.basename(folder.rstrip('/'))} | {step/1e6:.2f}M steps | {'stochastic' if a.stochastic else 'deterministic'}"
    for ep in range(a.episodes):
        o, _ = env.reset(seed=a.seed0 + ep); t, vs = 0, []
        while True:
            o, r, te, tr, info = env.step(act(o)); t += 1
            vx = info.get("x_velocity", float(env.unwrapped.data.qvel[0])); vs.append(vx)
            frames.append(caption(env.render(), [label, f"episode {ep+1}/{a.episodes}  t={t}  vx={vx:+.2f} m/s"]))
            if te or tr: break
        d = env.unwrapped.data; z, up = float(d.qpos[2]), float(d.xmat[1][8])
        why = "survived 1000 steps" if tr else ("height floor" if z < 0.33 else ("tilt" if up < 0.85 else "other"))
        manifest.append(dict(run=os.path.basename(folder.rstrip("/")), step=step, episode=ep + 1, reset_seed=a.seed0 + ep,
                             length=t, mean_vx=round(float(np.mean(vs)), 3), ended_by=why, mode="stochastic" if a.stochastic else "deterministic"))
        print(f"{label} ep{ep+1}: {t} steps, mean vx {np.mean(vs):+.2f}, ended by {why}")
    path = os.path.join(a.out, f"{os.path.basename(folder.rstrip('/'))}_step{step}.mp4")
    imageio.mimsave(path, frames, fps=fps); print("saved", path)

if manifest:
    mp = os.path.join(a.out, "manifest.csv"); new = not os.path.exists(mp)
    with open(mp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); new and w.writeheader(); w.writerows(manifest)
    print("manifest:", mp)
env.close()
