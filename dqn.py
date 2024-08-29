#!usr/bin/env python3 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Initialization of TensorBoard writer for visualization 
writer = SummaryWriter(log_dir='./runs/DQN_CartPole')

# **Hyperparameters 
BATCH_SIZE = 64 # number of experiences (state, action, reward, next state) sampled from the replay memory
LR = 0.001 # learning rate 
GAMMA = 0.99 # Discount factor for calculating return
TARGET_REPLACE_ITER = 100 # for update the target network 
MEMORY_CAPACITY = 1000000 # size of the data base 

##--FOR EXPLORATION--###
EPSILON_START = 1.0    #
EPSILON_END = 0.01     #
EPSILON_DECAY = 0.995  #
########################

EPISODES = 1000 # number of episodes 
MAX_STEPS = 200 # number of steps per episodes 

# Design the Q-network : 

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create of data-base (replay_buffer) for saving experiences  
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

#Create the DQN class agent 

class DQN:
    def __init__(self, n_states, n_actions):
        # Initialization 
        self.eval_net = Net(n_states, n_actions)
        self.target_net = Net(n_states, n_actions)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.steps_done = 0
        self.epsilon = EPSILON_START

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
    
    # Create the policy 
    def choose_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        elif isinstance(state, list):
            state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state type: {type(state)}")
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(env.action_space.n)
        else:
            with torch.no_grad():
                q_values = self.eval_net(state)
                return q_values.argmax().item()
            
    #this function is designed to store the transition information 
    # (state, action, reward, next_state, done) into a memory buffer

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.memory.add(transition)

    # Learning cyclem of DQN algorithm 
    def learn(self):

        # Checking replay memory 
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sampling from Replay Memory
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Converting Samples to Tensors 
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Calculating Q-Values and Target Values
        q_values = self.eval_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).detach()
        target_q_values = rewards + GAMMA * next_q_values.max(1)[0].unsqueeze(1) * (1 - dones)
        
        # Calculating LOSS function 
        loss = self.loss_func(q_values, target_q_values)
        
        # Update weights 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Updating the Target Network  
        '''Note : you can use other method for updating target network like Soft update '''
        if self.steps_done % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # Update epsilon for exploration 
        ''' you can use other method for exploration '''
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps_done += 1

    # Logging Loss and Q-Values to TensorBoard 
        writer.add_scalar('avg. Q', q_values.mean().item(), self.steps_done)
        writer.add_scalar('max. Q', q_values.max().item(), self.steps_done)
        writer.add_scalar('Loss', loss.item(), self.steps_done)
    
    # Saving and Loading the Model 
    def save(self, filename, directory):
        torch.save(self.eval_net.state_dict(), '%s/%s_dqn.pth' % (directory, filename))

    def load(self, filename, directory):
        self.eval_net.load_state_dict(torch.load('%s/%s_dqn.pth' % (directory, filename)))

# Main function for training cycle :

if __name__ == '__main__':
    
    file_name = "carte_pole"
    save_model = True
    load_model = False
    env = gym.make('CartPole-v1', render_mode="human")

    # Create the network storage folders
    #if not os.path.exists("./results"):
    #    os.makedirs("./results")
    #if save_model and not os.path.exists("./pytorch_models"):
    #    os.makedirs("./pytorch_models")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    dqn = DQN(n_states, n_actions)
    rewards = []

    if load_model:
        try:
            dqn.load(file_name, "./pytorch_models")
            print("loaded seccessflly")
        except:
            print(
                "Could not load the stored model parameters, initializing training with random parameters"
            ) 

    plt.ion() 
    fig, ax = plt.subplots()

    for episode in range(EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0
        done = False

        for step in range(MAX_STEPS):
            env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            dqn.store_transition(state, action, reward, next_state, done)
            dqn.learn()

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

        if save_model and episode % 100 == 0:
            dqn.save(file_name, directory="./pytorch_models")

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {episode_reward}, Epsilon: {dqn.epsilon:.3f}')

        writer.add_scalar('Episode/Reward', episode_reward, episode)

        ax.clear()
        ax.plot(rewards)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('DQN on CartPole-v1')
        plt.pause(0.01)  

    plt.ioff()  
    plt.show()

    env.close()
    writer.close()  
