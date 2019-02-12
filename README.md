# Pytorch-A2C

This Implementation of an Actor-Critic network gets an Image as input. The features of the image are shared bewteen the actor and 
the critic. Features of the image are extracted by two sequential convolution layers followed by a dense Layer. Actor and critic 
are getting the output of the dense layer as inputs.

Original version is from 
https://github.com/higgsfield/Imagination-Augmented-Agents

requiered Libraries:

* Pytorch 0.4.1 or later
* Torchvision
* gym

########################### Pacman training ###########################

Datails of the environment:
The code was written by @sracaniere from DeepMind
https://github.com/sracaniere

See https://arxiv.org/abs/1707.06203 [p.15ff] "C MiniPacman additional details" for more information about
about MiniPacman environment.

To start the training of the agent, run A2C_training_pacman.py. To see the result, run PlayPacman.py. 

The training method of this A2C is n-Step training. To 
prevent the agent from overfitting, at least 16 environments should be used to guarantee enough variation of training data. If a
different environment with more sparce rewards is used, the number of environments should be increased.


########################### KeyCollect training ###########################

In https://arxiv.org/abs/1707.06203 the Sokoban environment is described as second training environment. An implementation 
of this Sokoban environment can be found here: https://github.com/mpSchrader/gym-sokoban . To make this environment usable 
for the I2A it has to be downscaled to (3x80x80).

Since the generation of Sokoban puzzles can take from 2 seconds to over one minute and the A2C is synchronized, training would 
take way to long. A3C should be considered to use for this environment as the processes don´t have to wait for other processes to
finish level generation. The network proposed in the paper for Sokoban is larger and takes many more training epochs (
see https://arxiv.org/abs/1707.06203 for details) 

KeyCollect is a computationally very light weight environment that has sparse rewards and keeps training costs low. Traing of 
KeyCollect environment is done with the same network as MiniPacman. In KeyCollect the agent must reach one to five subgoals 
before reaching the goal. If he reaches the goal before, he is trapped in it until maximum steps of the level are reached

rewards:
every step: -1
subgoal:		1
goal:				10

To start the training of the agent, run A2C_training_keyCollect.py. To see the result, run PlayKeyCollect.py. 

The training method of this A2C is n-Step training. The advatage of n-Step training is that no target networks are reqiered. Due
to very sparse rewards to prevent the agent from overfitting, at least 32 environments should be used to guarantee enough variation 
of training data.  


Feel free to give any feedback or comments.
