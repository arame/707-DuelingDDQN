import gym
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env
from config import Config

Config.print_settings()
    
env = make_env('PongNoFrameskip-v4')
best_score = -np.inf
load_checkpoint = False
n_games = 500
agent = DuelingDDQNAgent(input_dims=(env.observation_space.shape), n_actions=env.action_space.n)

if load_checkpoint:
    agent.load_models()

n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(Config.n_games):
    done = False
    observation = env.reset()

    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward

        if not load_checkpoint:
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
        observation = observation_
        n_steps += 1
    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    print('episode: ', i,'score: ', score,
            ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
        'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    if avg_score > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.epsilon)
    if load_checkpoint and n_steps >= 18000:
        break

x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores, eps_history)
