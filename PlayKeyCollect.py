import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pickle

from common.multiprocessing_env import SubprocVecEnv

from common.actor_critic import ActorCritic
from common.Key_Collect import Key_Collect

plt.ion()

agentPath = 'actor_critic_KeyCollect'

USE_CUDA = torch.cuda.is_available()

def make_cuda(input):
    if USE_CUDA:
        return input.cuda()
    return input

num_envs = 1
# init plot
image = np.zeros((10, 10,3))
fig, ax = plt.subplots()
im = ax.imshow(image)
    

def displayImage(image, step, reward, value):              
    title = "step: {0} reward: {1:.2f} Value: {2:.2f}".format(step, reward, value[0][0])
    plt.title(title)        
    im.set_data(image)
    fig.canvas.draw()
    plt.pause(0.1)
 
num_keys = 1

def upscale(input):
    shape = list(input.shape)
    shape[-1] = 80
    shape[-2] = 80
    upscaled = np.zeros((shape))    
    for ij in np.ndindex(upscaled.shape[-2:]):        
        i,j=ij
        upscaled[...,i,j] = input[...,i//8,j//8]
    return upscaled

if __name__ == '__main__': 
    
    # init environment
    env = Key_Collect(max_steps=50, num_keys=num_keys)
    
    # load model
    actor_critic = ActorCritic((3,10,10), env.action_space.n)
    pretrained_dict = torch.load(agentPath, map_location='cpu')
    actor_critic.load_state_dict(pretrained_dict)

    actor_critic = make_cuda(actor_critic)

    # init game
    done = False
    state = env.reset()
    
    step = 1
    total_reward = 0
    
    while True:    
        current_state = torch.FloatTensor(state)
        
        action = actor_critic.act(make_cuda(current_state.unsqueeze(0)))        
        
        next_state, reward, done, _ = env.step(action.data[0][0])                
        total_reward += reward
        state = next_state

        _, value = actor_critic(make_cuda(current_state.unsqueeze(0)))        
        value = value.data.cpu().numpy()    
        
        image = torch.FloatTensor(upscale(state)).permute(1,2,0).cpu().numpy()                
                        
        displayImage(image, step, total_reward, value)
        step += 1
        if done:
            total_reward = 0
            state = env.reset()
            step = 1
        
