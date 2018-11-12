import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pickle

from IPython.display import clear_output

from common.minipacman import MiniPacman
from common.actor_critic import ActorCritic

plt.ion()


MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
mode = MODES[0]
USE_CUDA = torch.cuda.is_available()

# init plot
image = np.zeros((15, 19,3))
fig, ax = plt.subplots()
im = ax.imshow(image)
    

def displayImage(image, step, reward, value):
    clear_output(True)
    title = "step " + str(step) + " reward: " + str(reward) + " Value: " + str(value[0][0])        
    plt.title(title)    
    im.set_data(image)
    fig.canvas.draw()
    plt.pause(0.1)
    
# init environment
env = MiniPacman(mode=mode, frame_cap=1000)

# load model
agentPath = "actor_critic_pacman_" + mode
actor_critic = ActorCritic(env.observation_space.shape, env.action_space.n)
pretrained_dict = torch.load(agentPath)
actor_critic.load_state_dict(pretrained_dict)

if USE_CUDA:    
    actor_critic = actor_critic.cuda()



# init game
done = False
state = env.reset()
total_reward = 0
step = 1

#while not done:
while True:    
    current_state = torch.FloatTensor(state).unsqueeze(0)
    if USE_CUDA:
        current_state = current_state.cuda()
   
    action = actor_critic.act(current_state)
    next_state, reward, done, _ = env.step(action.data[0,0])
    total_reward += reward
    state = next_state

    _, value = actor_critic(current_state)
    value = value.data.cpu().numpy()    
    
    image = torch.FloatTensor(state).permute(1,2,0).cpu().numpy()
    displayImage(image, step, total_reward, value)
    step += 1
    if done:
        state = env.reset()
        step = 1
        total_reward = 0
