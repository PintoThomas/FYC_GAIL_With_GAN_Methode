import numpy as np
import tensorflow as tf


def normalize_action(expert_actions):
    expert_actions = expert_actions.tolist()
    for i in range(len(expert_actions)):
        if expert_actions[i] == 0:
            expert_actions[i] = np.array([0, 1], dtype=np.float32)
        elif expert_actions[i] == 1:
            expert_actions[i] = np.array([1, 0], dtype=np.float32)
    expert_actions = np.vstack(expert_actions)

    return expert_actions


def sample_buffer(data_state, data_action, batch_size):
    max_mem = 6500
    start = np.random.randint(0, 6500 - batch_size)
    batch = np.random.choice(max_mem, batch_size, replace=False)
    #batch =

    states = data_state[batch]
    actions = data_action[batch]

    return states, actions


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)
