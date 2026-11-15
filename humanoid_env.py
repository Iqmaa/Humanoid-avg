"""ONE definition of the robot task, shared by every algorithm (SB3 PPO, AVG, ...)."""
import os
import numpy as np
import gymnasium as gym

class UprightTermination(gym.Wrapper):
    """End the episode if the pelvis tilts more than ~30 degrees from vertical."""
    def __init__(self, env, min_up=0.85):
        super().__init__(env); self.min_up = min_up
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.data.xmat[1][8] < self.min_up:
            terminated = True
        return obs, r, terminated, truncated, info

class HeadingPenalty(gym.Wrapper):
    """Penalize facing away from +x (sin of the yaw angle, squared)."""
    def __init__(self, env, weight=2.0):
        super().__init__(env); self.weight = weight
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        fwd = -self.env.unwrapped.data.xmat[1].reshape(3, 3)[:, 1]
        return obs, r - self.weight * fwd[1] ** 2, terminated, truncated, info

class VelocityReward(gym.Wrapper):
    """Capped forward-velocity reward (replaces Humanoid's uncapped 1.25*vx term).
    Standing still ~ +1/step (alive bonus); walking at >=1 m/s ~ +5/step."""
    def __init__(self, env, weight=4.0, v_min=-0.5, v_max=1.0):
        super().__init__(env)
        self.weight, self.v_min, self.v_max = weight, v_min, v_max
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        vx = info.get("x_velocity", float(self.env.unwrapped.data.qvel[0]))
        r += self.weight * float(np.clip(vx, self.v_min, self.v_max))
        return obs, r, terminated, truncated, info

def make_robot_env(xml, **extra):
    env = gym.make(
        "Humanoid-v5", xml_file=os.path.abspath(xml), healthy_z_range=(0.33, 0.70),
        contact_cost_weight=0.0,
        healthy_reward=1.0,          # was 5.0 default: made standing still worth ~5/step
        forward_reward_weight=0.0,   # replaced by the capped VelocityReward wrapper
        # compact observation (qpos[2:] + qvel = 43 numbers):
        include_cinert_in_observation=False, include_cvel_in_observation=False,
        include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False,
        **extra)
    return HeadingPenalty(UprightTermination(VelocityReward(env)))

