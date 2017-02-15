# KerasCog
Models for cognitive neural processes using Keras.io

## Recent Updates:

Fixed the step() function to implement leakiness and randomness in neurons

## Current Objectives:


Check that randomness/leak is successfully being applied to neurons at each timestep. 

Compare the efficiency of training in keras to pycog (see sgd first)

Finish implementing Dale's law and applying to flip-flop


**Keep in mind whenever making a model to make sure that the time-scale of the noise and exponential decay have a large enough time constant relative to the timescale set by the individual steps in the RNN's evolution.

## Citation

This package is inspired by the pycog package of F. Song et al, built by the group of [Xiao-Jing Wang at New York University](http://www.cns.nyu.edu/wanglab/). For more information, take a look at:

* Song, H. F.\*, Yang, G. R.\*, & Wang, X.-J. "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework." *PLoS Comp. Bio.* 12, e1004792 (2016). (\* = equal contribution)

This package relies on Keras.io, a high-level neural networks library for Python:

* Chollet, François, "Keras", *Github*, https://github.com/fchollet/keras  

For more information on Keras, see the official website:

* https://keras.io/
