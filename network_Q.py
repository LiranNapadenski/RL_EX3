import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# TODO: define network, loss and optimiser(use learning rate of 0.1).
LR = 0.0081
input_dim = 16
output_dim = 4

frozenNet = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False))
optimizer = torch.optim.Adam(lr=LR, params=frozenNet.parameters())
loss_fn = nn.MSELoss(reduction='sum')

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        # TODO: Implement Step 1
        one_hot = torch.zeros(1, input_dim)
        one_hot[0, s] = 1
        Q = frozenNet(one_hot)
        action = torch.argmax(Q).item()
        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            action = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(action)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        # TODO: Implement Step 4

        one_hot = torch.zeros(1, input_dim)
        one_hot[0, s1] = 1
        Q1 = frozenNet(one_hot)

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        # TODO: Implement Step 5
        Q_target = Q.clone().detach()
        Q_target[0, action] = r + y * torch.max(Q1).item()

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        # TODO: Implement Step 6
        optimizer.zero_grad()
        loss = loss_fn(Q, Q_target)
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
