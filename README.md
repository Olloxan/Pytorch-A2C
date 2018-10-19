# Pytorch-A2C

This Implementation of an Actor-Critic network gets an Image as input. The features of the image are shared bewteen the actor and 
the critic. Features of the image are extracted by two sequential convolution layers followed by a dense Layer. Actor and critic 
are getting the output of the dense layer as inputs.

Original version is from 
https://github.com/higgsfield/Imagination-Augmented-Agents

requiered Libraries:

-Pytorch 0.4.1 or higher
-Torchvision
-gym

To start the training of the agent, run A2C_training_pacman.py. To see the result, run PlayPacman.py. 

The training method of this A2C is n-Step training. The advatage of n-Step training is that no target networks are reqiered. To 
prevent the agent from overfitting, at least 16 environments should be used to guarantee enough variation of training data. If a
different environment with more sparce rewards is used, the number of environments should be increased.
