import gym
import numpy as np
from Modele import GAIL
import tensorflow as tf

def train(envName):
    env = gym.make(envName)
    Agent = GAIL(env)
    env.reset()
    max_score = 0

    for i in range(500):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = np.argmax(tf.squeeze(Agent.choose_action(obs)))
            obs, rewards, done, info = env.step(action)
            score += rewards
            #env.render()
        if score >= max_score:
            print("Model Save")
            max_score = score
            Agent.save_model()

        print(f'{i} : Le score est de {score}')
        Agent.train()

if __name__ == "__main__":
    train('CartPole-v1')
    #train('BipedalWalker-v3')



