#!/usr/bin/env python

#import libraries 
import networkx as nx
import numpy as np
from tqdm import trange
import math
from itertools import product
import os
import multiprocessing as mp

#Class to run and save the simulatios
class ARM_MM():
    def __init__(self, params, iters, seed, savehist=True):
        defaults = {'B' : [0.25], 'XM' : [0.0], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [500000]}
        plist = [params[p] if p in params else defaults[p] for p in defaults]
        self.params = list(product(*plist))
        self.iters = iters 
        self.rng = np.random.default_rng(seed)
        self.savehist = savehist
        
    #Create intial opinions with the Mass Media opinion as the last node
    def initializing(self, N, XM):
        config = np.zeros(N)
        config[N-1] = XM
        for i in np.arange(N-1):   
        #initial Gaussian distribution
            while True: 
                config[i] = self.rng.normal(0.5, 0.2)
                if 0 <= config[i] and config[i] <= 1:
                    break
        config = config.reshape(-1, 1)
        init_config = config
        return config

    #Create complete network with MM
    def complete_graph_MM(self, N, config):
        G=nx.complete_graph(N)
        for i in G.nodes:          
             G.add_nodes_from([i], opinion=config[i])
        return G

    #Create ring network with MM
    def circulantMM(self, N, config):         
        G=nx.circulant_graph(N, [1])      
        for i in G.nodes:        
            G.add_nodes_from([i], opinion=config[i])   
        #Including Mass Media  
        G.add_nodes_from([N-1], opinion=config[N-1]) 
        #Add connections between the MM and all the nodes 
        for i in G.nodes:
            if i!=N-1:      
                G.add_edge(N-1, i)
        return G
    
   #Small-world network including Mass Media
    def small_world_MM(self, N, config):
        G=nx.watts_strogatz_graph(N - 1, 4, 0.3)
        contador=0
        for i in G.nodes:
            G.nodes[i]['opinion'] = config[i]
            contador=contador+1
        #Including Mass Media
        G.add_nodes_from([N-1], opinion=config[N-1])
        for i in G.nodes:
            if i!=N-1:
                G.add_edge(N-1, i)
        return G

    #Save the data in a .txt file 
    def save_data(self, G, iters, step, directory_name):
        if step == 0:
            with open(f"./outputfolder/{directory_name}/history_iteration-{iters}.txt", "w") as f:
                for k in G.nodes:
                    f.write(str(k) + "\t") 
                f.write("\n")
        with open(f"./outputfolder/{directory_name}/history_iteration-{iters}.txt", "a") as f:
            for k in G.nodes:
                f.write("{:.6f}\t".format(G.nodes[k]["opinion"][0]))  
            f.write("\n")
   
    #Save the last 1000 steps 
    def asymptotic_data(self, G, iters, step, S, directory_name):   
        if step == S-1000:
            with open(f"./outputfolder/{directory_name}/asymp_steps_iter-{iters}.txt", "w") as f:
                for k in G.nodes:
                    f.write(str(k) + "\t") 
                f.write("\n")     
        with open(f"./outputfolder/{directory_name}/asymp_steps_iter-{iters}.txt", "a") as f:   
            for k in G.nodes:
                f.write("{:.6f}\t".format(G.nodes[k]["opinion"][0]))  
            f.write("\n")
        
    #Dynamics of the network 
    def arm_MM(self): 
        for param in self.params:   
            B, XM, N, E, T, R, S = param
            directory_name = "B_{:.2f}-XM_{:.2f}".format(round(B, 2), round(XM, 2))
            if not os.path.exists("./outputfolder/"+str(directory_name)):   
                os.makedirs("./outputfolder/" + str(directory_name))
            for it in range(self.iters):   
                config = self.initializing(N, XM)
                G = self.circulantMM(N,config)
                for step in trange(S, desc='Simulating interactions', disable=True):
                    #Choose a random node except mass media
                    i = self.rng.choice(np.delete(G.nodes,-1)) 
                    #Choose j as Mass Media with probability B
                    if self.rng.random() <= B: 
                        j = N-1
                    else:
                        #Choose a random neighbor with probability 1 - B
                        j = self.rng.choice(np.delete(G[i],len(G[i])-1)) 
                    #Calcualte distance between opinions
                    dist = (abs(G.nodes[i]["opinion"] - G.nodes[j]["opinion"])) 
                    #Calculate probability of interaction
                    prob = math.pow(0.5, dist/E)  
                    if self.rng.random() <= prob:
                        #Condition for atrarction d < T
                        if dist <= T:   
                            #i get closer to j, R times their distance
                            G.nodes[i]["opinion"] = G.nodes[i]["opinion"] + R * (G.nodes[j]["opinion"] - G.nodes[i]["opinion"])
                        else: 
                            #Condition for repulsion d > T
                            G.nodes[i]["opinion"] = G.nodes[i]["opinion"] - R * (G.nodes[j]["opinion"] - G.nodes[i]["opinion"])
                        #Set limits [0-1]        
                        G.nodes[i]["opinion"] = np.maximum(0, np.minimum(1,G.nodes[i]["opinion"])) 
                    #Save history each N steps if requested
                    if self.savehist == True and step%N == 0:
                        self.save_data(G, it, step, directory_name)
                    #Save last 1000 steps
                    if step >= S-1000: 
                         self.asymptotic_data(G, it, step, S, directory_name)
        return G                

def expA_grid(B = 0.5, XM = 0.5):
    params = {'B' : [B], 'XM' : [XM], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [2000000]}
    exp = ARM_MM(params, iters=20, seed=None, savehist=False)
    exp.arm_MM() 

if __name__ == "__main__":
    #Number of laptop CPUs to be used
    n_cpu = mp.cpu_count()
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of XM and B values
    B_range = np.arange(0, 1.0 + 1/50, 1/50)
    XM_range = np.arange(0, 1.0 + 1/50, 1/50)
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(B, XM) for B in B_range for XM in XM_range]
    # Call expC_grid for all parameter tuples using pool.map
    results = pool.starmap(expA_grid, param_tuples)
    # Close the pool
    pool.close()

