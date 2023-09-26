# Attraction-Repulsion Model with Mass Media

The Attraction-Repulsion Model with Mass Media is an agent-based model focused on opinion dynamics to study the polarization of a system. This model has been developed by Mateo Carpio and Mario Cosenza. 

# Getting Started

## Usage

The script can be run on a cluster or on a multi-core machine using the following command:

    python diffusionMPI.py -n <number_of_processes> -E <id_of_experiment> -R <random_seed>

where <number_of_processes> is the number of processes to be used in parallelization, <id_of_experiment> is the experiment wanted to run, and <random_seed> is the seed.

The output of the code is saved in the current and output folder directory, which is created if it does not exist.
