import gymnasium
import numpy as np
from typing import Tuple, List
import tensorflow as tf


class MyEnv(gymnasium.Env):
    def __init__(self, env_name: str):
        super(MyEnv, self).__init__()

        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        observation = self.env.reset()[0]
        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observation, reward, done, end, _ = self.env.step(action)  # ignore 'info' return
        return (observation.astype(np.float32),
                np.array(reward, np.float32),
                np.array(done, bool),
                np.array(end, bool))

    def tf_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action], [tf.float32, tf.float32, tf.bool, tf.bool])

    def render(self):
        return self.env.render()
