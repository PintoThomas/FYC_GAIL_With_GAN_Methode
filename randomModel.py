import random
import gym
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.reset()

    scores = []

    for i in range(20):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = random.randint(0, 1)
            obs, rewards, done, info = env.step(action)
            score += rewards
            env.render()
        print(f'{i} : Le score est de {score}')
        scores.append(score)
    print(np.mean(scores))
    plt.plot(scores)
    plt.show()