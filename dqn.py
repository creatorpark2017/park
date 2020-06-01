import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import random
import cv2
import collections
import gym
class CNN(Module):
    def __init__(self, input_shape, n_actions):
        super(CNN, self).__init__()
        
        self.cnn_layers = Sequential(
        Conv2d(input_shape[0], 64, kernel_size = 8, stride = 4  ),
        ReLU(inplace=True),
        Conv2d(64,64, kernel_size = 3, stride =1),
        ReLU(inplace = True))
        
        conv_out_size = self.get_conv_out(input_shape)
        self.linear_layers = Sequential(
        Linear(conv_out_size, 512),
        nn.ReLU(),
        Linear(512, n_actions))
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def get_conv_out(self,shape):
        o = self.cnn_layers(torch.zeros(1,*shape))
        return int(np.prod(o.size()))
    
    
    def forward(self, x):
        x = self.cnn_layers(x).view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x    
    
class mlp(Module):
    def __init__(self, input_shape, n_actions):
        super(mlp, self).__init__()
        
        self.linear_layers = Sequential(
        Linear(input_shape[0], 128),
        nn.ReLU(),
        Linear(128, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0001)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self,x):
        return self.linear_layers(x)
    
    
class Replay():
    def __init__(self, maxmemory, batch_size):

        self.memory = deque(maxlen = maxmemory)
        self.batch_size = batch_size
    def store(self,state, action, reward, nextstate, done):
        self.memory.append([state,action,reward,nextstate,done])
        
    def sample(self):
        #return random.sample(self.memory, len(self.memory))
        return self.memory
    def samplevalues(self):
        random_samples = self.sample()
        states = []
        actions = []
        rewards = []
        nextstates = []
        dones=[]
        for i in range(len(random_samples)):
            nextstates.append(random_samples[i][3])
        for i in range(len(random_samples)):
            states.append(random_samples[i][0])
        for i in range(len(random_samples)):
            actions.append(random_samples[i][1])
        for i in range(len(random_samples)):
            rewards.append(random_samples[i][2])
        for i in range(len(random_samples)):
            dones.append(random_samples[i][4])
        
        rand_indices = np.random.permutation(len(random_samples))[:self.batch_size]
        next_state = np.asarray(nextstates)[rand_indices]
        state = np.asarray(states)[rand_indices]
        reward = np.asarray(rewards)[rand_indices]
        action = np.asarray(actions)[rand_indices]
        done = np.asarray(dones)[rand_indices]
        done_index = np.where(done)[0].tolist()
        
        return random_samples, state, action, reward, next_state, done_index
        
        
class dqagent():
    def __init__(self, nn, maxmemory,batch_size, Replay, env,decay, nn_update, gamma):
              
        self.epsilon = 0.99
        self.actionspace = env.action_space.n
        self.statespace = env.observation_space.shape
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n
        self.nn = nn(input_shape, n_actions)
        self.nn2 = nn(input_shape, n_actions)
        self.decay = decay
        self.gamma = gamma
        self.replay = Replay(maxmemory,batch_size)
        self.nn_update = nn_update
        self.batch_size = batch_size
        self.counter = 0
    def action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.actionspace)])
        else:
            state = torch.tensor([state], dtype = torch.float).to(self.nn.device)
            Q = self.nn.forward(state)
            action = torch.argmax(Q).item()
        return action
    def decrement_epsilon(self):
        self.epsilon *= self.decay if self.epsilon > 0.1 else 0.1

    def update_value(self):
        self.counter +=1
        self.decrement_epsilon()
        random_samples, state, action, reward, nextstate, done_index = self.replay.samplevalues()
        self.next = nextstate
        if len(random_samples) < 50000:
            return
        self.randomsamples = random_samples
        state = torch.tensor(state, dtype = torch.float).to(self.nn.device)
        reward = torch.tensor(reward, dtype = torch.float).to(self.nn.device)
        nextstate = torch.tensor(nextstate, dtype = torch.float).to(self.nn.device)
        
        
        if self.counter % self.nn_update == 0:
            self.nn2.load_state_dict(self.nn.state_dict())
        
        Q_pred = self.nn.forward(state)[:, action]
        Q_max = self.nn2.forward(nextstate).max(dim=1)[0]
        Q_max[done_index] = 0
        Q_target = reward + self.gamma * Q_max
        
        self.nn.optimizer.zero_grad()
       
        loss = self.nn.loss(Q_target, Q_pred).to(self.nn.device)
        loss.backward()
        self.nn.optimizer.step()
        
        
            
