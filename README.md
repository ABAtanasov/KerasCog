# TensorCog
Models for cognitive neural processes using Google's Tensorflow

## Recent Updates:

Changed over to tensorflow from Keras 

## Current Objectives:

Change Networks.py to build a leak recurrent neural network with dale's law enforced

**Keep in mind whenever making a model to make sure that the time-scale of the noise and exponential decay have a large enough time constant relative to the timescale set by the individual steps in the RNN's evolution.

## Citation

This package is inspired by the pycog package of F. Song et al, built by the group of [Xiao-Jing Wang at New York University](http://www.cns.nyu.edu/wanglab/). For more information, take a look at:

* Song, H. F.\*, Yang, G. R.\*, & Wang, X.-J. "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework." *PLoS Comp. Bio.* 12, e1004792 (2016). (\* = equal contribution)
