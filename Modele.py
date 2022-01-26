import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from Utils import *


class Generator(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()

        self.dense_1 = Dense(state_dim, activation='relu')
        self.dense_2 = Dense(50, activation='relu')
        self.dense_3 = Dense(50, activation='relu')
        self.dense_4 = Dense(action_dim, activation="softmax")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.inputDim = state_dim + action_dim

        self.dense_1 = Dense(self.inputDim, activation="relu")
        self.dense_2 = Dense(50, activation="relu")
        self.dense_3 = Dense(50, activation="relu")
        self.dense_4 = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x


class GAIL:
    def __init__(self, env):
        self.env = env
        self.expert_state = np.genfromtxt('./trajectory/observations.csv')
        self.expert_actions = normalize_action(np.genfromtxt('./trajectory/actions.csv'))

        state_dim = len(env.observation_space.high)
        if self.env.unwrapped.spec.id in ["CartPole-v1"]:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]

        self.generator = Generator(state_dim, action_dim)
        self.discriminator = Discriminator(state_dim, action_dim)

        self.gen_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.disc_optimizer = tf.keras.optimizers.Adam(0.00001)

    def choose_action(self, states):
        state = np.array([states])
        actions = self.generator(state)
        return actions

    def train(self):
        steps = 200
        total_gen_loss = 0
        total_disc_loss = 0

        for i in range(steps):
            batch_exp_states, batch_exp_actions = sample_buffer(self.expert_state, self.expert_actions, 100)
            batch_states, _ = sample_buffer(self.expert_state, self.expert_actions, 100)
            #batch_states = batch_exp_states

            with tf.GradientTape(persistent=True) as tape:
                actions = tf.squeeze(self.choose_action(batch_states))

                expert_probs = self.discriminator(batch_exp_states, batch_exp_actions)
                policy_probs = self.discriminator(batch_states, actions)

                disc_loss = discriminator_loss(BinaryCrossentropy(from_logits=True), expert_probs, policy_probs)
                gen_loss = generator_loss(BinaryCrossentropy(from_logits=True), policy_probs)


            grad_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            grad_gen = tape.gradient(gen_loss, self.generator.trainable_variables)

            self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
            self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print(f'Gen Loss : {total_gen_loss / 200} , Disc Loss : {total_disc_loss / 200}\n')

    def save_model(self):
        self.generator.save_weights(f'./model/{self.env.unwrapped.spec.id}_generator.h5')
        #self.discriminator.save_weights('./model/discriminator.h5')
