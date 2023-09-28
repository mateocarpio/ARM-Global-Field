# Attraction-Repulsion Model (ARM) with Mass Media

The Attraction-Repulsion Model with Mass Media is an agent-based model focused on opinion dynamics to study the polarization of a system. This model has been developed by Mateo Carpio and Mario Cosenza. 

## 1) exp_arm_mm.py

This file contains the main code for running the ARM with mass media. The code uses the multiprocessing library to parallelize the code across multiple processors.
The code has been written in Python and requires the following packages to be installed:
   
    numpy
    networkx
    multiprocessing
    math
    os
    itertools
    argparse
    
The script is organized into two Classes. The first one is ARM_MM which could run the simulation for different kinds of networks and repeat over iterations. The class could save data in two ways, the whole history of the nodes' opinions or only its asymptotic behavior. The data is saved in a text file. For the asymptotic behavior it saves the last one thousand steps, while for the whole history, the data is saved for every N (number of nodes) steps.  
## Usage

The script can be run on a cluster or on a multi-core machine using the following command:

    python exp_arm_mm.py -n <number_of_processes> -E <id_of_experiment> -R <random_seed>

where <number_of_processes> is the number of processes to be used in parallelization, <id_of_experiment> is the experiment wanted to run, and <random_seed> is the seed for the random number generator. The experiment can be found at the final of the script. 

The output of the code is saved in the current and output folder directory, which is created if it does not exist.

## 2) Getting_Results_ARM_MM.ipynb
