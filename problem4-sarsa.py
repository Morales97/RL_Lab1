# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Initial author: [Alessio Russo - alessior@kth.se]
# Adapted by: Valetnin Minoz, Daniel Morales

# %reset -fs
# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import pickle as pkl
import pdb

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, lb, ub):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - lb) / (ub - lb)
    return x

def fbasis_func(eta, s):
    ''' To Avoid writing out eq below
    '''
    assert eta.shape[1] == s.shape[0]
    return np.cos(np.pi*np.matmul(eta,s))


# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
state = env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Constants
p = 2                       # fourier basis p
n = state.size              # size of s

# Parameters
discount_factor = 1.        # Value of gamma
alpha = 0.1                # learning rate
eligibility_t_p = 0.01     # eligibility trace parameter
reduce_alpha = True         # Whether or not alpha is to be scaled down
momentum = 0.2           # SGD with momentum, set to 0 for normal SGD
scale_basis = True          # Whether or not basis is to be scaled

# Params for running
N_episodes = 200            # Number of episodes to run for training
verb = True
show = False

# Important tensors
# eta = np.array([[1,0], [0,1], [2,1], [1,2], [2,0], [0,2], [0,0], [1,1], [2,2]]) # 130
#eta = np.array([[2,0], [0,2], [1,1]]) # 140
# eta = np.array([[1,0], [0,1]]) # 200
# eta = np.array([[2,1], [1,2]])  # 200, not a single episode successful
# eta = np.array([[1,1]]) # 180
#eta = np.array([[1,0], [0,1], [1,1]])   # 200, not a single episode successful
#eta = np.array([[1,0], [0,2], [1,1]])  # 200, not a single episode successful
#eta = np.array([[2,1], [1,2], [2,2], [1,0], [0,1], [0,2], [2,0]]) # 120
eta = np.array([[1,0], [0,1], [2,1], [1,2], [2,0], [0,2], [1,1], [2,2]]) # 120
#eta = np.array([[i,j] for i in range(p+1) for j in range(p+1)]) # 130

m_basis = eta.shape[0]
W = np.zeros((k,m_basis))   # Weight matrix

# Scaling the fourier basis
if scale_basis:
    norm_eta = np.linalg.norm(eta,2,1)
    norm_eta[norm_eta == 0] = 1 # if ||eta_i||=0 then alpha_i=alpha
    alpha = np.divide(alpha, norm_eta)

pdb.set_trace()

# Training process
episode_reward_list = []    # Used to save episodes reward
vals = []
reduce_counter = 0
for i in range(N_episodes):
    if verb and not i % (N_episodes//100): print(f'ep:{i}', end = '\r')
    # Reset enviroment data
    done = False
    state = scale_state_variables(env.reset(), low, high)
    total_episode_reward = 0
    z = np.zeros_like(W)
    velocity = np.zeros_like(W)

    phi = fbasis_func(eta, state)
    Q = np.dot(W, phi)
    vals.append(np.max(Q))

    while not done:
        if show and (i % 25 == 0): env.render()
        # generate Q(s,a) and produce action
        phi = fbasis_func(eta, state)
        Q = np.dot(W, phi)
        action = np.argmax(Q)

        # Get next state, reward and done state
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state, low, high)

        # Update eligibility trace
        z *= eligibility_t_p * discount_factor
        z[action] += phi # gradient of Q_w(s,a) w.r.t. w_a is
        z = np.clip(z, -5, 5)

        # Get Q_w(s[t+1],a[t+1]) and next action
        phi = fbasis_func(eta, next_state)
        next_Q = np.dot(W, phi)
        next_action = np.argmax(next_Q)

        # Update Weights
        delta_t = reward + discount_factor*next_Q[next_action] - Q[action]
        if scale_basis:
            velocity = momentum*velocity + delta_t*np.matmul(z,np.diag(alpha))
        else:
            velocity = momentum*velocity + delta_t*alpha*z
        W += velocity

        # Update episode reward and state for next iteration
        total_episode_reward += reward
        state = next_state

    if reduce_alpha and  total_episode_reward > -200:
        # If win, scale alpha by .8 or .6 if good win
        reduce_counter += 1
        alpha *= 0.8 - 0.2*(total_episode_reward > -130)




    # Append episode reward and close env
    episode_reward_list.append(total_episode_reward)
    env.close()
if verb: print(f'ep:{N_episodes}')
print('Done.')

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')

#plt.title('Total Reward vs Episodes')
#plt.legend()
#plt.grid(alpha=0.3)
#plt.show()
#plt.plot(vals)
plt.show()

if input('input anything to save weights\n'):
    with open(f'weights.pkl','wb') as f:
        pkl.dump({'W':W,'N':eta}, f)
    print('saved.')
else: print('discarded.')
