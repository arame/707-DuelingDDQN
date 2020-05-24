import numpy as np
import torch as T
from network import Network
from replay_memory import ReplayBuffer
from config import Config

class Agent(object):
    def __init__(self, n_actions, input_dims):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.epsilon = Config.epsilon
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(input_dims, n_actions)
        name_root = Config.env_name+'_'+ Config.algo
        self.q_eval = Network(self.n_actions, input_dims=self.input_dims, name=name_root + '_q_eval')
        self.q_next = Network(self.n_actions, input_dims=self.input_dims, name=name_root + '_q_next')

    def store_transition(self, state, action, reward, state_new, done):
        self.memory.store_transition(state, action, reward, state_new, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(Config.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)

            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if Config.replace_target_cnt is not None and self.learn_step_counter % Config.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decay_epsilon(self):
        new_value = self.epsilon * Config.eps_decay
        if new_value > Config.eps_min:
            self.epsilon = new_value

    def learn(self):
        if self.memory.mem_cntr < Config.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_new, dones = self.sample_memory()
        indices = np.arange(Config.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_new, A_s_new = self.q_next.forward(states_new)
        V_s_eval, A_s_eval = self.q_eval.forward(states_new)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_new, (A_s_new - A_s_new.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + Config.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
