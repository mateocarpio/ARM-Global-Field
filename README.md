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
    


## Usage

The script can be run on a cluster or on a multi-core machine using the following command:

    python exp_arm_mm.py -n <number_of_processes> -E <id_of_experiment> -R <random_seed>

where <number_of_processes> is the number of processes to be used in parallelization, <id_of_experiment> is the experiment wanted to run, and <random_seed> is the seed for the random number generator. The experiment can be found at the final of the script. 

The script is organized into two Classes. The first one is ARM_MM which could run the simulation for different kinds of networks and repeat over iterations. The class could save data in two ways, the whole history of the nodes' opinions or only its asymptotic behavior. The data is saved in a text file. For the asymptotic behavior, it saves the last one thousand steps, while for the whole history, the data is saved for every N (number of nodes) steps.  
The second class is called process_data, and it allows us to make all the statistics. The statistics are saved in two folders. "Asymptotic_statistics" saves the asymptotic statistical parameters for each iteration of a simulation. "history_statistics" saves the statistical parameter for each N steps. 


## 2) Getting_Results_ARM_MM.ipynb

In this notebook, we plot the results of the experiments. We can find the results for the asymptotic statistical quantities as a function of $x_m$ and $B$ for the global, ring and small-world networks including mass media. 
