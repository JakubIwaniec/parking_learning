import tensorflow as tf
import tqdm
import collections
import statistics
from typing import Tuple
from figure import plot_figure
from discount import discount_rewards
from Env import Env
from MyEnv import MyEnv

from PIL import Image
import webbrowser
import os
frames = []
episodes_per_GIF = 600
want_gifs = True  # nalezy ustawic domyslna aplikacje do gifow
# (np. Internet Explorer) wtedy gify beda pojawialy się w jednym okienku


HIDDEN_UNITS_SIZE = 64  # ilość neuronów w warstwie ukrytej
EPISODES_AMOUNT = 3000
MAX_STEPS_PER_EPISODE = 500
MIN_EPISODES_CRITERION = 100  # zmienna wykorzystywana przy obliczaniu średniej

GAMMA = 0.99
LEARNING_RATE = 0.01


class NeuralNetwork(tf.keras.Model):
    def __init__(self, out_size: int, hidden_units: int):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        self.common = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.out = tf.keras.layers.Dense(out_size)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.common(inputs)
        x = self.out(x)
        return x


#@tf.function
def run_episode(initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    observations = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    observation = initial_state
    obs_shape = initial_state.shape

    for step in tf.range(MAX_STEPS_PER_EPISODE):
        net_output = net(tf.expand_dims(observation, 0))
        action = tf.random.categorical(net_output, 1)[0, 0]
        next_observation, reward, done, _ = env.tf_step(action)
        next_observation.set_shape(obs_shape)

        observations = observations.write(step, observation)
        actions = actions.write(step, action)
        rewards = rewards.write(step, reward)

        observation = next_observation
        if done:
            break

        if want_gifs and (episode_no + 1) % episodes_per_GIF == 0:
            frames.append(Image.fromarray(env.render()))

    observations = observations.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    return observations, actions, rewards


def compute_loss(actions, action_probabilities, returns):
    actions = tf.expand_dims(actions, axis=1)
    chosen_action_probs = tf.gather_nd(action_probabilities, actions, batch_dims=1)
    chosen_action_probs = tf.math.log(chosen_action_probs)
    chosen_action_probs = tf.cast(chosen_action_probs, dtype=tf.float32)
    loss = -tf.reduce_sum(chosen_action_probs * returns)
    return loss


#@tf.function
def train_step(initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    observations, actions, rewards = run_episode(initial_state)
    with tf.GradientTape() as tape:
        net_output = net(observations)
        action_probabilities = tf.nn.softmax(net_output)
        returns = discount_rewards(rewards, GAMMA, True)
        for i in range(len(actions)):
            print(net_output[i], '---', action_probabilities[i])
        loss = compute_loss(actions, action_probabilities, returns)

    grads = tape.gradient(loss, net.trainable_variables)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))

    return tf.math.reduce_sum(rewards), tf.size(rewards)

#env = MyEnv("CartPole-v1")
env = Env()
net = NeuralNetwork(env.action_space.n, HIDDEN_UNITS_SIZE)

episodes_reward = collections.deque(maxlen=MIN_EPISODES_CRITERION)
episodes_steps = collections.deque(maxlen=MIN_EPISODES_CRITERION)
rewards = []
steps = []

with tqdm.trange(EPISODES_AMOUNT) as learning:
    for episode_no in learning:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward, steps_amount = train_step(initial_state)
        episode_reward = int(episode_reward)
        steps_amount = int(steps_amount)
        episodes_reward.append(episode_reward)
        episodes_steps.append(steps_amount)
        running_reward = statistics.mean(episodes_reward)
        running_steps = statistics.mean(episodes_steps)
        rewards.append(running_reward)
        steps.append(running_steps)
        learning.set_description(f'Episode {episode_no}')
        learning.set_postfix(reward=episode_reward, steps=steps_amount,
                             last_rewards_mean=running_reward, last_steps_mean=running_steps)
        if want_gifs and (episode_no + 1) % episodes_per_GIF == 0:
            animation_path = f'animation.gif'
            frames[0].save(animation_path, save_all=True, append_images=frames[1:], loop=0, duration=1)

            # Otwarcie animacji w przeglądarce
            webbrowser.open('file://' + os.path.realpath(animation_path))
            frames = []


plot_figure(rewards, steps, 'REINFORCE')
env.close()
