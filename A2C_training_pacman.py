import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from common.multiprocessing_env import SubprocVecEnv
from common.minipacman import MiniPacman
from common.actor_critic import ActorCritic, RolloutStorage
from common.logger import Logger
from common.myTimer import myTimer


logger = Logger()
timer = myTimer()

USE_CUDA = torch.cuda.is_available()

MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
mode = MODES[0]
num_envs = 16

def make_env():   
    def _thunk():
        env = MiniPacman(mode, 1000)        
        return env
    return _thunk

if __name__ == '__main__': 
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)    

    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n
        
    #a2c hyperparams:
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 5
    num_frames = int(1e6)

    #rmsprop hyperparams:
    lr    = 7e-4
    eps   = 1e-5
    alpha = 0.99

    #Init a2c and rmsprop
    actor_critic = ActorCritic(state_shape, num_actions)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    
    if USE_CUDA:
        actor_critic = actor_critic.cuda()



    rollout = RolloutStorage(num_steps, num_envs, state_shape)
    if USE_CUDA:        
        rollout.cuda()

    all_rewards = []
    all_losses  = []    
    all_step_scores = []

    state = envs.reset() 
    
    state = torch.FloatTensor(np.float32(state))
    if USE_CUDA:
        state = state.cuda()

    rollout.states[0].copy_(state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)    

    timer.update(time.time())

    for i_update in range(num_frames):

        for step in range(num_steps):                             
            action = actor_critic.act(state)
                               
            next_state, reward, finished, _ = envs.step(action.squeeze(1).cpu().data.numpy())
               
            reward = torch.FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            finished_masks = torch.FloatTensor(1-np.array(finished)).unsqueeze(1)                                                       

            final_rewards *= finished_masks
            final_rewards += (1-finished_masks) * episode_rewards                       
                                                              
            episode_rewards *= finished_masks
            state = torch.FloatTensor(np.float32(next_state))                                               
            
            if USE_CUDA:
                finished_masks = finished_masks.cuda()
                state = state.cuda()
            
            rollout.insert(step, state, action.data, reward, finished_masks)


        _, next_value = actor_critic(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape),
            rollout.actions.view(-1, 1)
        )

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)        
        advantages = returns - values

        value_loss = advantages.pow(2).mean()        
        action_loss = -(advantages.data * action_log_probs).mean()

        optimizer.zero_grad()        
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        optimizer.step()
    
        if i_update % 100 == 0:            
            all_rewards.append(final_rewards.mean())
            all_losses.append(loss.item())
            timer.update(time.time())
            loopstogo = (num_frames - i_update) / 100
            estimatedtimetogo = timer.getTimeToGo(loopstogo)
            
            print('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))            
            print('loss %s' % all_losses[-1])
            logger.printDayFormat("estimated time to run: ", estimatedtimetogo)
            print("######## AC_Pacman_{0} ########".format(mode))                        
        rollout.after_update()
        
    logger.log(all_rewards, "Data/", "all_rewards_{0}.txt".format(mode))  
    logger.log(all_losses, "Data/", "all_losses_{0}.txt".format(mode))      
    logger.log_state_dict(actor_critic.state_dict(), "Data/actor_critic_{0}".format(mode))    

    

