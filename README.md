# Multi-Layer-Perceptron-CNN-Architecture-Robotics

This repo contains my implementation of a Multi-Layer Perceptron Convolutional Neural Network which was trained on a MuJoCo-based simulation dataset as well as a dataset full of real-life robotic manipulator data, with the end goal being to generate a trajectory for pushing a bowl and a second trajectory for picking up a block and placing it in a bowl. The code used, as well as loss curves and other documentation relevant to these tasks can be found in this repo. 

Ultimately, in this project I developed 4 separate models based on the same fundamental architecture (with slight alterations for different situations) with two of these models being trained on different simulations situations, and the other 2 being trained on those same situations in real life. While I saw decent success with my simulated models, the real-life data based models were not as effective. This can mostly be traced to the limitations of Behavior-cloning models such as the one used in this project, though these issues can likely also be traced to issues properly interfacing with the utilized robotic arm.

The below images contain the loss curves for each trained model, with the first two models being trained on a simple bowl-pushing scenario and the second two models being trained on a pick-and-place scenario involving a bowl and cube.

## Sim-Data Push Model:

![Model 1 Sim](/Deliverables/Model1/sim_push_loss_curve.png)

## Real-Data Push Model:

![Model 1 Real](/Deliverables/Model1_Real/real_push_loss_curve.png)

## Sim-Data Pick-and-Place Model:

![Model 2 Sim](/Deliverables/Model2/new_sim_pick_place_loss_curve.png)

## Real-Data Pick-and-Place Model:

![Model 2 Real](/Deliverables/Model2_Real/new_real_pick_place_loss_curve.png)

## Analysis

All 4 loss curves followed a fairly expected trajectory, steadily decreasing as the number of epochs increased. Notice too that the Pick-and-Place models both required more epochs to undergo sufficient loss decreases.

The architecture developed in this project used PyTorch as its main library, while numpy and Scikit were used for various minor calculations and other aspects of trajectory generation. All algorithms were developed in Python.
