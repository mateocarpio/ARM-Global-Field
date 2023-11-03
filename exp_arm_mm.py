#!/usr/bin/env python

import networkx as nx
import numpy as np
from tqdm import trange
import math
from itertools import product
import os
import argparse
import multiprocessing as mp

#Class to run and save the simulatios
class ARM_MM():
    def __init__(self, params, iters, seed, savehist, network):
        defaults = {'B' : [0.25], 'XM' : [0.0], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [500000]}
        plist = [params[p] if p in params else defaults[p] for p in defaults]
        self.params = list(product(*plist))
        self.iters = iters 
        self.rng = np.random.default_rng(seed)
        self.savehist = savehist
        self.network = network
        
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
            with open(f"./{self.network}/outputfolder/{directory_name}/history_iteration-{iters}.txt", "w") as f:
                for k in G.nodes:
                    f.write(str(k) + "\t") 
                f.write("\n")
        with open(f"./{self.network}/outputfolder/{directory_name}/history_iteration-{iters}.txt", "a") as f:
            for k in G.nodes:
                f.write("{:.6f}\t".format(G.nodes[k]["opinion"][0]))  
            f.write("\n")
   
    #Save the last 1000 steps 
    def asymptotic_data(self, G, iters, step, S, directory_name):   
        if step == S-1000:
            with open(f"./{self.network}/outputfolder/{directory_name}/asymp_steps_iter-{iters}.txt", "w") as f:
                for k in G.nodes:
                    f.write(str(k) + "\t") 
                f.write("\n")     
        with open(f"./{self.network}/outputfolder/{directory_name}/asymp_steps_iter-{iters}.txt", "a") as f:   
            for k in G.nodes:
                f.write("{:.6f}\t".format(G.nodes[k]["opinion"][0]))  
            f.write("\n")
        
    #Dynamics of the network 
    def arm_MM(self): 
        for param in self.params:   
            B, XM, N, E, T, R, S = param
            directory_name = "B_{:.2f}-XM_{:.2f}".format(round(B, 2), round(XM, 2))
            if not os.path.exists(f"./{self.network}/outputfolder/"+str(directory_name)):   
                os.makedirs(f"./{self.network}/outputfolder/" + str(directory_name))
            #For iteration of the same simulation 
            for it in range(self.iters):   
                config = self.initializing(N, XM)
                if self.network == "global":
                    G = self.complete_graph_MM(N,config)
                elif self.network == "ring":
                    G = self.circulantMM(N,config)
                elif self.network == "small-world":
                    G = self.small_world_MM(N,config)
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

class process_data():
    
    def __init__(self, B, XM, iters, hist_saved, network):

        self.B = B
        
        self.XM = XM

        self.iters = iters 
        
        self.directory_name = "B_{:.2f}-XM_{:.2f}".format(round(B, 2), round(XM, 2))
        
        self.hist_saved = hist_saved

        self.network = network

        if self.hist_saved == True:
            self.file_path1 = f"./{self.network}/outputfolder/"+ str(self.directory_name) + f"/history_iteration-{self.iters-1}.txt"
            self.data = np.loadtxt(self.file_path1, skiprows=1)
        
        delta_bins = 1/50
        self.bins = np.arange(-delta_bins/2,1+delta_bins,delta_bins)

    #Identify the size of the maximum group and where it is located 
    def maximum_group(self, frequency):
        
        m = max(frequency)
        indexes=[i for i, j in enumerate(frequency) if j == m]
        
        return m,indexes

    #Get the variance, the maximum group and its difference with the MM
    def get_statistical_parameters(self, config):
        
        frequency, bins = np.histogram(config,bins = self.bins)

        S_max,indexes = self.maximum_group(frequency)

        variance = np.var(config)

        S_M = frequency[np.digitize(self.XM, bins)-1]

        Delta_M = S_max - S_M

        return S_max, variance, Delta_M

    # Plot the statistical parameters 
    def save_stat(self):
        
        file_path = f"./{self.network}/history_statistics/" + "statistical_parameters_B_{:.2f}-XM_{:.2f}.txt".format(round(self.B, 2), round(self.XM, 2))
        
        if not os.path.exists(f"./{self.network}/history_statistics"):
            
            os.makedirs(f"./{self.network}/history_statistics")
        
        with open(file_path, "w") as f:
            
            f.write("Step \t S_max  \t variance \t Delta_M \n")
            
        for i in range(0,len(self.data),10):
            
            S_max, variance, Delta_M = self.get_statistical_parameters(self.data[i,:-1])

            with open(file_path, "a") as f:

                f.write("{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(i*100, S_max, variance, Delta_M))

    #Get the variance, the maximum group and its difference with the MM
    def get_asymptotical_parameters(self, iters):
        
        file_path = f"./{self.network}/outputfolder/" + str(self.directory_name) + f"/asymp_steps_iter-{iters}.txt"
            
        data = np.loadtxt(file_path, skiprows=1)
            
        S_max, variance, Delta_M = 0, 0, 0
            
        len_data = len(data)
        
        for i in range(len_data):
            
            new_S_max, new_variance, new_Delta_M= self.get_statistical_parameters(data[i,:-1])
                
            S_max, variance, Delta_M = S_max+new_S_max, variance + new_variance, Delta_M + new_Delta_M
        
        return S_max/len_data, variance/len_data, Delta_M/len_data

    
    def prom_iters(self):
        
        prom_S_max, prom_variance, prom_Delta_M = 0, 0, 0 
        
        for it in range(self.iters): 
            
            iter_S_max, iter_variance, iter_Delta_M = self.get_asymptotical_parameters(it)
            
            prom_S_max, prom_variance, prom_Delta_M = prom_S_max + iter_S_max, prom_variance + iter_variance, prom_Delta_M + iter_Delta_M
            
        return prom_S_max/self.iters, prom_variance/self.iters, prom_Delta_M/self.iters

    def save_prom_parameters(self):
        
        file_path = f"./{self.network}/Asymptotic_statistics/" + "Asypm_statistical_parameters_B_{:.2f}-XM_{:.2f}.txt".format(round(self.B, 2), round(self.XM, 2))
        
        if not os.path.exists(f"./{self.network}/Asymptotic_statistics"):
            
            os.makedirs(f"./{self.network}/Asymptotic_statistics")
        
        with open(file_path, "w") as f:
            
            f.write("Iteration \t S_max  \t sigma \t Delta_M \t \n")
            
        prom_S_max, prom_variance, prom_Delta_M = 0, 0, 0 

        for it in range(self.iters): 
            
            iter_S_max, iter_variance, iter_Delta_M = self.get_asymptotical_parameters(it)
            
            with open(file_path, "a") as f:
                
                f.write("{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(it, iter_S_max, iter_variance,  iter_Delta_M))
                
            prom_S_max, prom_variance, prom_Delta_M = prom_S_max + iter_S_max, prom_variance + iter_variance, prom_Delta_M + iter_Delta_M
        
        with open(file_path, "a") as f:
            
            f.write("average\t{:.6f}\t{:.6f}\t{:.6f}\n".format(prom_S_max/self.iters, prom_variance/self.iters,  prom_Delta_M/self.iters))
            

    #Plot the histograms 
    def save_hist_plot_agents(self, config, step):

        # Set up the histogram
        fig = plt.figure(figsize = (12,8))
        ax = plt.gca()


        # Define the number of bins and the color map
        num_bins = len(self.bins)
        color_map = plt.cm.get_cmap('coolwarm')
        norm = mcolors.Normalize(vmin=0, vmax=num_bins-1)

        # Create the histogram
        n, bins, patches = plt.hist(config, self.bins, edgecolor='black')

        # Set the color of each patch based on its position in the histogram
        for i in range(len(patches)):
          
            if (i == np.digitize(self.XM, self.bins)-1): #Find the index where XM is located
                color = 'gold' # Set the middle bin to yellow
            else:
                color = color_map(norm(i))

            patches[i].set_facecolor(color)

        # Set the axis labels and title
        ax.set_xlabel('Opinion Position', size = 20)
        ax.set_ylabel('Total Agents', size = 20)

        plt.xlim([-0.02, 1.02])
        plt.ylim([0, 101])

        ax.set_title('Step %d' %step, size = 20)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 18)

        dummy_data = np.random.normal(0, 1, 0)

        plt.hist(dummy_data, bins=num_bins, alpha=0.5, color='gold', label = f' $X_M=${round(self.XM,2)}')

        plt.legend(loc='upper right', fontsize = 18 )
        
        if not os.path.exists("./outputfolder/"+str(self.directory_name)+"/figures"):
            os.makedirs("./outputfolder/"+str(self.directory_name)+"/figures")
        
        plt.savefig("./outputfolder/"+ str(self.directory_name) +"/figures/{:03d}.png".format(step))

        plt.close()
        
    
    def create_images(self):
        for i in range(0,len(self.data),10):
            self.save_hist_plot_agents(self.data[i,:-1], i*100)
    
    
    def create_gif(self):
        
        images_in = "./outputfolder/"+ str(self.directory_name) +"/figures/****.png"
        gif_image_out = "./outputfolder/"+ str(self.directory_name) +"/animation_hist.gif"
        imgs = (Image.open(f) for f in Tcl().call('lsort', '-dict', glob.glob(images_in)))
        img = next(imgs)
        img.save(fp = gif_image_out, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)
    

def run_hystory_global(B = 0.5, XM = 0.5):
    params = {'B' : [B], 'XM' : [XM], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [2000000]}
    exp = ARM_MM(params, iters=1, seed=None, savehist=True, network="global")
    exp.arm_MM()
    results = process_data(B = B, XM = XM, iters=1, hist_saved=True, network="global")
    results.save_stat()
#Reproduce Figures 3.3 and 3.4
def exp_history_global_xm_05(seed=None, n_cpu=2):
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of XM and B values
    B_range = np.arange(0, 0.80, 0.1)
    XM_range = [0.50]
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(B, XM) for B in B_range for XM in XM_range]
    # Call run_hystory_global for all parameter tuples using pool.map
    results = pool.starmap(run_hystory_global, param_tuples)
    # Close the pool
    pool.close()
#Reproduce Figure 3.5
def exp_history_global_xm_00(seed=None, n_cpu=2):
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of XM and B values
    B_range = np.arange(0, 0.8, 0.1)
    XM_range = [0.50]
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(B, XM) for B in B_range for XM in XM_range]
    # Call run_hystory_global for all parameter tuples using pool.map
    results = pool.starmap(run_hystory_global, param_tuples)
    # Close the pool
    pool.close()

#Reproduce Figure 3.6 and 3.7
def run_asymptotic_global(B, XM):
    params = {'B' : [B], 'XM' : [XM], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [2000000]}
    exp = ARM_MM(params, iters=2, seed=None, savehist=False, network="global")
    exp.arm_MM() 
    results = process_data(XM=XM, B=B, iters=2, hist_saved=False, network="global")
    results.save_prom_parameters()
def exp_grid_xm_B_global(seed=None, n_cpu=2):
    #Number of laptop CPUs to be used
    # n_cpu = mp.cpu_count()
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of XM and B values
    B_range = np.arange(0, 1.0 + 1/50, 1/50)
    XM_range = np.arange(0, 1.0 + 1/50, 1/50)
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(B, XM) for B in B_range for XM in XM_range]
    # Call expC_grid for all parameter tuples using pool.map
    results = pool.starmap(run_asymptotic_global, param_tuples)
    # Close the pool
    pool.close()

#Reproduce Figure 3.8
def run_asymptotic_ring(B, XM):
    params = {'B' : [B], 'XM' : [XM], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [2000000]}
    exp = ARM_MM(params, iters=2, seed=None, savehist=False, network="ring")
    exp.arm_MM() 
    results = process_data(XM=XM, B=B, iters=2, hist_saved=False, network="ring")
    results.save_prom_parameters()
def exp_grid_xm_B_ring(seed=None, n_cpu=2):
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of XM and B values
    B_range = np.arange(0, 1.0 + 1/50, 1/50)
    XM_range = np.arange(0, 1.0 + 1/50, 1/50)
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(B, XM) for B in B_range for XM in XM_range]
    # Call expC_grid for all parameter tuples using pool.map
    results = pool.starmap(run_asymptotic_ring, param_tuples)
    # Close the pool
    pool.close()

#Reproduce Figure 3.9
def run_asymptotic_SW(B, XM):
    params = {'B' : [B], 'XM' : [XM], 'N' : [101], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [2000000]}
    exp = ARM_MM(params, iters=2, seed=None, savehist=False, network="small-world")
    exp.arm_MM() 
    results = process_data(XM=XM, B=B, iters=2, hist_saved=False, network="small-world")
    results.save_prom_parameters()
def exp_grid_xm_B_SW(seed=None, n_cpu=2):
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of B and X_M values
    B_range = np.arange(0, 1.0 + 1/50, 1/50)
    XM_range = np.arange(0, 1.0 + 1/50, 1/50)
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(B, XM) for B in B_range for XM in XM_range]
    # Call run_asymptotic_SW for all parameter tuples using pool.map
    results = pool.starmap(run_asymptotic_SW, param_tuples)
    # Close the pool
    pool.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-n', '--ns', type=int, default=2, \
        help="-takes the number of processes to be used in parallelization",)
    parser.add_argument('-E', '--exps', type=str, nargs='+', required=True, \
                        help='IDs of experiments to run')
    parser.add_argument('-R', '--rand_seed', type=int, default=None, \
                        help='Seed for random number generation')
    args = parser.parse_args()
    # Run selected experiments.
    exps = {'history_global' : exp_history_global, 'grid_xm_B_global' : exp_grid_xm_B_global, 'grid_xm_B_ring' : exp_grid_xm_B_ring, \
        'grid_xm_B_sw' : exp_grid_xm_B_SW}

    for id in args.exps:
        exps[id](args.rand_seed,args.ns)



