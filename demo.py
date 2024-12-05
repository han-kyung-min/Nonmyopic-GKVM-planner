#!/usr/bin/env python
# coding: utf-8

# # A greeting from Chenxi! This is a tutorial for the method introduced in our paper: "Nonmyopic Informative Path Planning Based on Global Kriging Variance Minimization." IEEE Robotics and Automation Letters (2022).
#
#

# ## Prerequisite: Install Dependencies (code has been tested on a Google colab GPU runtime)

# In[1]:


get_ipython().system('pip install tsp_solver2==0.4.1 # other versions might not be compatible.')
get_ipython().system('pip install gpytorch==1.5.0')

# In[2]:

import numpy as np
import gpytorch
import pickle
from tsp_solver.greedy import solve_tsp
from ippmpe import MPE_k, create_gpr,TSP,norm
import matplotlib.pyplot as plt
import cv2
import utils
import sys

# 1. load the map
def load_map(filename):
    with open(filename, 'rb') as handle:
        a = pickle.load(handle)
        return a["observe2d"], a["uncertainty2d"]
def main(argv):
    #filename = 'data/corner_.pgm'
    #filename = 'data/corridor_.pgm'
    #filename = 'data/loop_.pgm'
    #filename = 'data/loop_with_corridor_.pgm'      # <=== doesn't work with the default config's kernel size
    #filename = 'data/room_.pgm'
    #filename = 'data/room_with_corner_.pgm'

    if(len(sys.argv) != 2):
        print("num sys argv %d"%len(sys.argv))
        print("usage: %s <config file path>" % sys.argv[0])
        return -1
    filename = sys.argv[1]
    #  map_visited, init_uncertainty = load_map("data/observations.pickle")
    map_visited, init_uncertainty = utils.load_gkvm_map_from_gmapimg( filename,
                                                                    ukn = 205, free = 254, occ = 0, num_ds =1 )

    # fig, ax= plt.subplots(1,2)
    # ax[0].imshow(map_visited)
    # ax[1].imshow(init_uncertainty)
    # plt.show()

    nrows, ncols = map_visited.shape
    nsize = max(nrows, ncols)
    # 2. covert it to point cloud observations
    X = np.argwhere(map_visited>-0.1)[:,[1,0]]
    y = np.zeros(len(X))
    #print(X.shape, y.shape)

    # ## 2. Create GP model to represent uncertainty, and do MPE sampling.
    # In[4]:
    num_samples = 100

    config = {
        "kernel_scale": 8,   # increase kernel scale if img size is greater than 100x100
        "kernel_noise": 0.0001,
    }
    gpr = create_gpr(X, y, config = config)
    X,Y = np.meshgrid(np.arange(nsize),np.arange(nsize))
    mesh = np.vstack([X.ravel()[None,:], Y.ravel()[None,:]]).T
    mpe_samples = MPE_k(gpr, mesh, k=num_samples)  # num sample is a variable

    # ## 3. solving TSP problem based on MPE landmarks

    # In[7]:

    # add the initial location of the robot
    init_pos = mpe_samples[0] #np.array([1,1])
    landmarks = np.vstack([init_pos[None,:], mpe_samples])

    # connectivity matrix. This could be simplified by a kNN version for a large scale problem.
    D = np.zeros((len(landmarks), len(landmarks)))
    for i in range(len(landmarks)):
        for j in range(len(landmarks)):
            D[i,j] = norm(landmarks[i]-landmarks[j])

    # TSP solver
    path = solve_tsp( D, endpoints=(0,None) )
    print(path)

    path_len = 0.0
    plt.imshow(init_uncertainty, origin='lower')
    plt.plot( init_pos[0], init_pos[1], 'bo', markersize=10)
    for i in range(len(path)-1):
        x1,y1 = landmarks[path[i]]
        x2,y2 = landmarks[path[i+1]]
        seg_len = ( (x2-x1)**2 + (y2-y1)**2 ) ** 0.5
        path_len += seg_len
        plt.plot([x1,x2],[y1,y2], c="r")
    plt.scatter(landmarks[1:,0], landmarks[1:,1], c="m")
    plt.scatter(landmarks[0,0], landmarks[0,1], c="r", s=40)
    plt.show()

    print("num samples: %d  path len: %f" % (num_samples, path_len) )

if __name__ == '__main__':
   main(sys.argv)