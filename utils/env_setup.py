import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

class FireReset(gym.Wrapper):
    """
    TEST Wrapper som automatiskt trycker FIRE (action=3) efter reset()
    för att hoppa förbi startskärmen i Space Invaders.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(2):
            obs, _, terminated, truncated, info = self.env.step(3)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

def create_env(render_mode="rgb_array"):
    """
    Skapar en SpaceInvaders‐miljö med:
      1) FireReset (hoppar över startskärm)
      2) AtariPreprocessing (gråskala + nedskalning)
      3) FrameStack (4 stackade frames)
    """
    env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode=render_mode)
    env = FireReset(env)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, 4)
    return env
