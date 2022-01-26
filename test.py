import gym
import numpy as np
from Modele import GAIL
import tensorflow as tf
import matplotlib.pyplot as plt


def test(envName):
    env = gym.make(envName)
    Agent = GAIL(env)
    env.reset()

    for i in range(1):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = np.argmax(tf.squeeze(Agent.choose_action(obs)))
            obs, rewards, done, info = env.step(action)
            score += rewards
            env.render()
        print(f'{i} : Le score est de {score}')


    Agent.generator.load_weights('./model/CartPole-v1_generator1.h5')

    scores = []

    for i in range(20):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = np.argmax(tf.squeeze(Agent.choose_action(obs)))
            obs, rewards, done, info = env.step(action)
            score += rewards
            env.render()
        print(f'{i} : Le score est de {score}')
        scores.append(score)

    return scores

if __name__ == "__main__":
    scoresCartpole = test('CartPole-v1')
    print(f"Score Moyen Random on Cartpole : {np.mean(scoresCartpole)}")
    plt.plot(scoresCartpole)
    plt.show()

    # scoresWalker = test('Walker-v1')
    # print(f"Score Moyen Random on Cartpole : {np.mean(scoresCartpole)}")
    # plt.plot(scoresCartpole)
    # plt.show()
