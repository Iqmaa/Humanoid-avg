import argparse, os, torch
import numpy as np
import pandas as pd
import gymnasium as gym
from humanoid_env import make_robot_env
from avg import AVG

class SimpleCfg:
    pass

def evaluate(method, model_path, xml="humanoid2.xml", num_episodes=50, out_csv="eval_results.csv"):
    env = make_robot_env(xml=xml)
    eval_seeds = [5000 + i for i in range(num_episodes)]
    records = []

    if method == "AVG":
        cfg = SimpleCfg()
        cfg.obs_dim = env.observation_space.shape[0]
        cfg.action_dim = env.action_space.shape[0]
        cfg.device = torch.device("cpu")
        cfg.actor_lr = 1e-4
        cfg.critic_lr = 1e-4
        cfg.betas = [0.0, 0.999]
        cfg.alpha_lr = 0.07
        cfg.gamma = 0.99
        cfg.nhid_actor = 256
        cfg.nhid_critic = 256
        
        agent = AVG(cfg)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.actor.eval()
        
        obs_mean = ckpt.get("obs_rms_mean", None)
        obs_var = ckpt.get("obs_rms_var", None)
        def normalize(o):
            if obs_mean is not None and obs_var is not None:
                return (o - obs_mean) / np.sqrt(obs_var + 1e-8)
            return o

        def get_action(o):
            # Must mirror Actor.forward in avg.py: phi is L2-normalised before mu().
            with torch.no_grad():
                x = torch.as_tensor(normalize(o), dtype=torch.float32).unsqueeze(0)
                phi = agent.actor.phi(x)
                phi = phi / torch.norm(phi, dim=1, keepdim=True)
                a = agent.actor.mu(phi).tanh().cpu().numpy().squeeze(0)
            # Training env used ClipAction (ctrlrange +-0.4); do the same here.
            return np.clip(a, env.action_space.low, env.action_space.high)

    elif method == "PPO":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # Locate matching VecNormalize stats file
        vecnorm_path = model_path.replace(".zip", "").replace("model_", "model_vecnormalize_") + ".pkl"
        if not os.path.exists(vecnorm_path):
            vecnorm_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")

        vec_env = DummyVecEnv([lambda: make_robot_env(xml=xml)])
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"[Eval] Loaded VecNormalize stats from {vecnorm_path}")
        else:
            print(f"[Warning] No VecNormalize stats found at {vecnorm_path}!")

        model = PPO.load(model_path, env=vec_env)
        def get_action(o):
            norm_o = vec_env.normalize_obs(o)
            return model.predict(norm_o, deterministic=True)[0]

    for ep, seed in enumerate(eval_seeds):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_ret, ep_len = 0.0, 0
        velocities, heights = [], []

        while not done:
            action = get_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            
            # Extract forward velocity and torso Z height directly from MuJoCo if missing in info
            vel = info.get("x_velocity", float(env.unwrapped.data.qvel[0]))
            ht = info.get("z_height", float(env.unwrapped.data.qpos[2]))
            velocities.append(vel)
            heights.append(ht)
            done = term or trunc

        mean_vel = float(np.mean(velocities)) if velocities else 0.0
        mean_ht = float(np.mean(heights)) if heights else 0.0
        walked = (ep_len == 1000) and (mean_vel >= 0.8) and (mean_ht >= 0.35)

        records.append({
            "episode": ep + 1,
            "seed": seed,
            "return": ep_ret,
            "length": ep_len,
            "mean_forward_velocity": mean_vel,
            "mean_torso_height": mean_ht,
            "fell": ep_len < 1000,
            "walks_criteria_passed": walked
        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    print(f"Mean Return: {df['return'].mean():.2f} ± {df['return'].std():.2f}")
    print(f"Mean Length: {df['length'].mean():.1f} ± {df['length'].std():.1f}")
    print(f"Mean Velocity: {df['mean_forward_velocity'].mean():.3f} m/s")
    print(f"Mean Height: {df['mean_torso_height'].mean():.3f} m")
    print(f"Success Rate (Walks): {df['walks_criteria_passed'].mean() * 100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["AVG", "PPO"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="eval_results.csv")
    args = parser.parse_args()
    evaluate(args.method, args.model, out_csv=args.out)

