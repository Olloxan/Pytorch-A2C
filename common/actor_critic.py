import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def act(self, x, deterministic=False):       
        logit, value = self.forward(x)                       
        probs = F.softmax(logit, dim=1)       
        
        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(num_samples=1)
        
        return action
    
    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)
        
        probs     = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)
        
        action_log_probs = log_probs.gather(1, action)       

        entropy = -(probs * log_probs).sum(1).mean()
        
        return logit, action_log_probs, value, entropy
    

class ActorCritic(OnPolicy):
    def __init__(self, in_shape, num_actions):
        super(ActorCritic, self).__init__()
        
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = 16

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        fc_size = 256
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), fc_size),
            nn.ReLU(),
        )
                
        self.critic  = nn.Linear(fc_size, 1)
        self.actor   = nn.Linear(fc_size, num_actions)
        
    def forward(self, x):            
        x = self.features(x)        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def feature_size(self):       
        convoutput1 = self.calculate_conv_output(self.in_shape[1:3], self.out_channels, 3)       
        convoutput2 = self.calculate_conv_output(convoutput1[1:3], self.out_channels, 3, 2)
        features = int(np.prod(convoutput2))
        return features

    def calculate_conv_output(self, img_dim, out_channels, kernel_size, stride=1, padding=0):        
        output_width = (img_dim[0] - kernel_size + 2*padding) // stride + 1
        output_height = (img_dim[1] - kernel_size + 2*padding) // stride + 1
        return [out_channels, output_width, output_height]

    
class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, state_shape):
        self.num_steps = num_steps
        self.num_envs  = num_envs
        self.states  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.rewards = torch.zeros(num_steps,     num_envs, 1)
        self.masks   = torch.ones(num_steps  + 1, num_envs, 1)
        self.actions = torch.zeros(num_steps,     num_envs, 1).long()
        self.use_cuda = False
            
    def cuda(self):
        self.use_cuda  = True
        self.states    = self.states.cuda()
        self.rewards   = self.rewards.cuda()
        self.masks     = self.masks.cuda()
        self.actions   = self.actions.cuda()
        
    def insert(self, step, state, action, reward, mask):
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)
        
    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        
    def compute_returns(self, next_value, gamma):
        returns   = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        if self.use_cuda:
            returns = returns.cuda()
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]
