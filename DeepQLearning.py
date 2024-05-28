import tensorflow as tf
import tqdm
import collections
import statistics
from typing import Tuple
from figure import plot_figure
from discount import discount_rewards
from Env import Env
from MyEnv import MyEnv
import numpy as np

from PIL import Image
import webbrowser
import os
frames = []
episodes_per_GIF = 600
want_gifs = True  # nalezy ustawic domyslna aplikacje do gifow
# (np. Internet Explorer) wtedy gify beda pojawialy się w jednym okienku

HIDDEN_UNITS_SIZE = 64
EPISODES_AMOUNT = 6000
MAX_STEPS_PER_EPISODE = 500
MIN_EPISODES_CRITERION = 100

GAMMA = 0.99
LEARNING_RATE = 0.001

BUFFER_SIZE = 4000
BATCH_SIZE = 32
MIN_EPISODES_BEFORE_TRAIN = 50
DECREASE_EPSILON_TO_EPISODE = 4 / 5 * EPISODES_AMOUNT
EPSILON_MIN = 0.01


class NeuralNetwork(tf.keras.Model):
    def __init__(self, out_size: int, hidden_units: int):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE)

        self.common = tf.keras.layers.Dense(hidden_units, activation='elu')
        self.hidden = tf.keras.layers.Dense(hidden_units, activation='elu')
        self.out = tf.keras.layers.Dense(out_size)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.common(inputs)
        x = self.hidden(x)
        x = self.out(x)
        return x


#@tf.function
def epsilon_greedy_policy(observation, epsilon):
    random_number = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
    if random_number < epsilon:
        random_action = tf.random.uniform(shape=(), minval=0, maxval=action_space, dtype=tf.int32)
        return random_action
    else:
        q_values = net(tf.expand_dims(observation, 0))[0]
        return tf.cast(tf.argmax(q_values), dtype=tf.int32)


def run_episode(initial_state, epsilon):
    observation = initial_state
    episode_reward = 0.0
    steps = MAX_STEPS_PER_EPISODE
    for step in tf.range(MAX_STEPS_PER_EPISODE):
        action = epsilon_greedy_policy(observation, epsilon)
        next_observation, reward, done, _ = env.tf_step(action)
        steps_replay_buffer.append((observation, action, reward, done, next_observation))
        episode_reward += reward
        if done:
            steps = step
            break
        observation = next_observation

        if want_gifs and (episode_no + 1) % episodes_per_GIF == 0:
            frames.append(Image.fromarray(env.render()))

    return episode_reward, steps


def samples_from_batch(batch_size):
    indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(steps_replay_buffer), dtype=tf.int32)
    samples = [steps_replay_buffer[index] for index in indices]
    samples = [np.array(data_field) for data_field in zip(*samples)]
    return samples


@tf.function
def train_step(samples):
    observations, actions, rewards, done, next_observations = samples
    next_q_values = net(next_observations)
    next_q_values = tf.reduce_max(next_q_values, axis=1)

    if_not_last = 1.0 - float(done)

    target_q_values = rewards + if_not_last * GAMMA * next_q_values
    target_q_values = tf.reshape(target_q_values, (-1, 1))

    mask = tf.one_hot(actions, action_space)

    with tf.GradientTape() as tape:
        q_values = net(observations)
        q_values = tf.reduce_sum(q_values * mask, axis=1, keepdims=True)

        loss = tf.reduce_mean(loss_fn(q_values, target_q_values))

    grads = tape.gradient(loss, net.trainable_variables)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))


env = Env()
action_space = env.action_space.n
net = NeuralNetwork(action_space, hidden_units=HIDDEN_UNITS_SIZE)
loss_fn = tf.keras.losses.mean_squared_error

steps_replay_buffer = collections.deque(maxlen=BUFFER_SIZE)

episodes_reward = collections.deque(maxlen=MIN_EPISODES_CRITERION)
episodes_steps = collections.deque(maxlen=MIN_EPISODES_CRITERION)
rewards = []
steps = []

with tqdm.trange(EPISODES_AMOUNT) as learning:
    for episode_no in learning:
        epsilon = max(1 - episode_no / DECREASE_EPSILON_TO_EPISODE, EPSILON_MIN)
        episode_reward, steps_amount = run_episode(env.reset(), epsilon)
        if episode_no > MIN_EPISODES_BEFORE_TRAIN:
            train_step(samples_from_batch(BATCH_SIZE))
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


plot_figure(rewards, steps, 'Deep Q-Learning')
env.close()
