# Pytorch-A2C

Original version is from 
https://github.com/higgsfield/Imagination-Augmented-Agents

This Implementation of an Actor-Critic network gets an Image as input. The features of the image are shared bewteen the actor and 
the critic. Features of the image are extracted by two sequential convolution layers followed by a dense Layer. Actor and critic 
are getting the output of the dense layer as inputs.
