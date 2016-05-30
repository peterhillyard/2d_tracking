'''
Created on May 30, 2016

@author: Peter Hillyard
'''

# This script is used to evaluate the performance of the three antenna types
# used during our experimentation in Atlanta.

import sys
import numpy as np

sys.path.insert(0, '../my_span_package') # Adds higher directory to python modules path.
import rti_class as rti

# node_loc_fname = 'data/node_loc_data/neals_old_house/node_loc_2012_11_01.txt'
# rss_fname = 'data/rss_data/neals_old_house/rss_2012_11_01_all.txt'

# node_loc_fname = 'data/node_loc_data/airbnb_atl/node_loc_2016_05_17.txt'
# rss_fname = 'data/rss_data/airbnb_atl/rss_2016_05_17_omni_peter_2_all.txt'
# pivot_coords_fname = 'data/true_loc_data/airbnb_atl/pivot_coords_2016_05_17.txt'
# path_ind_fname = 'data/true_loc_data/airbnb_atl/pivot_idx_2016_05_17.txt'

node_loc_fname = 'data/node_loc_data/pizza_house/node_loc_2016_05_16.txt'
rss_fname = 'data/rss_data/pizza_house/rss_2016_05_18_epatch_peter_all.txt'
pivot_coords_fname = 'data/true_loc_data/pizza_house/pivot_coords_2016_05_16.txt'
path_ind_fname = 'data/true_loc_data/pizza_house/pivot_idx_2016_05_16.txt'

num_ch = 4
delta_p = 0.8
sigmax2 = 0.5
delta = 3.0
epl = 0.25
rti_T = 0.5
skip_time=16.0
cal_time=16.0

path_start_time = skip_time+cal_time
speed = 1.0 / 2.0

# Initialize rti stuff
rti_obj = rti.RTI(node_loc_fname,num_ch,delta_p,sigmax2,delta,epl,rti_T,skip_time,cal_time)
rti_obj.set_true_coord_params(path_start_time,speed,pivot_coords_fname,path_ind_fname)

est_coord_vec = []
true_coord_vec = []

# loop through all lines in the RSS .txt file
with open(rss_fname,'r') as f:
    for line in f:
        rti_obj.observe(line)
        #rti_obj.plot_current_image(pause_time=0.05)
        true_coord_vec.append(rti_obj.get_true_coord())
        est_coord_vec.append(rti_obj.get_est_coord())
        
true_coord_vec = np.array(true_coord_vec)
est_coord_vec = np.array(est_coord_vec)

print rti_obj.compute_rmse(true_coord_vec,est_coord_vec)
sre,yvals = rti_obj.compute_cdf(true_coord_vec,est_coord_vec)



