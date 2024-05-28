import tensorflow as tf
import numpy as np


def discount_rewards(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    """Discount rewards by gamma factor."""
    shape = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=shape)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)  # odwróć
    discounted_sum = tf.constant(0.0, dtype=tf.float32)
    discounted_sum_shape = discounted_sum.shape
    for index in tf.range(shape):
        reward = rewards[index]
        discounted_sum = tf.cast(reward, tf.float32) + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(index, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns)
                                                               + np.finfo(np.float32).eps.item()))
    return returns
