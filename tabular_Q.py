import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
NOISE = lambda x : 1/x
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0
    FOUND_GOAL = False
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        # 2. Get new state and reward from environment
        # 3. Update Q-Table with new knowledge
        # 4. Update total reward
        # 5. Update episode if we reached the Goal State
        action = 0
        if np.random.rand() < NOISE(j) or not FOUND_GOAL:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[s,:])
        
        ns, reward, terminated, _ = env.step(action)
        Q[s, action] = Q[s, action] + lr * (reward + y * Q[ns,:].max() - Q[s, action])
        
        s = ns
        rAll += reward
        if reward != 0 :
            FOUND_GOAL = True
        if terminated:
            break
    
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
