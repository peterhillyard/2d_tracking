'''
Created on May 31, 2016

@author: Peter Hillyard
'''

# This script is used to plot a link using all channels
import numpy as np
import matplotlib.pyplot as plt


fname = 'data/rss_data/pizza_house/rss_2016_05_18_epatch_peter_walk_all.txt'
N = 20
C = 4
L = N*(N-1)*C
link_database = np.zeros((L,4))
link_idx = 0

for ch in range(C):
    for tx in range(N):
        for rx in range(N):
            if tx == rx:
                continue
            link_database[link_idx,0] = link_idx
            link_database[link_idx,1] = tx
            link_database[link_idx,2] = rx
            link_database[link_idx,3] = ch
            link_idx += 1

plot_link=None
plot_pair = [1,2]

if plot_link is not None:
    link_to_plot = int(plot_link)
elif plot_pair is not None:
    tmp = (link_database[:,1] == plot_pair[0]-1) & (link_database[:,2] == plot_pair[1]-1)
    link_to_plot = link_database[tmp,0].astype('int')

timestamp_vec = []
rss_vec = []

with open(fname,'r') as f:
    for line in f:
        split_line = line.split(' ')
        
        timestamp_vec.append(float(split_line.pop()))
        rss = np.array([float(ii) for ii in split_line])
        
        rss_vec.append(rss[link_to_plot].tolist())
        
rss_vec = np.array(rss_vec)
rss_vec[rss_vec == 127] = np.nan
timestamp_vec = np.array(timestamp_vec)-timestamp_vec[0]

plt.plot(timestamp_vec,rss_vec)
plt.show()
        