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

loc = 'airbnb_atl'
node_loc_date = '2016_05_17'
rss_date_and_name = '2016_05_17_omni_amal_all'
pivot_coord_date = '2016_05_17'
pivot_idx_date = '2016_05_17'

# loc = 'pizza_house'
# node_loc_date = '2016_05_16'
# rss_date_and_name = '2016_05_20_lcom_peter_2_all'
# pivot_coord_date ='2016_05_16'
# pivot_idx_date = '2016_05_16'

node_loc_fname = 'data/node_loc_data/' + loc + '/node_loc_' + node_loc_date + '.txt'
rss_fname = 'data/rss_data/' + loc + '/rss_' + rss_date_and_name + '.txt'
pivot_coords_fname = 'data/true_loc_data/' + loc + '/pivot_coords_' + pivot_coord_date + '.txt'
path_ind_fname = 'data/true_loc_data/' + loc + '/pivot_idx_' + pivot_idx_date + '.txt'
results_fname = 'data/results/' + loc + '/performance_' + rss_date_and_name + '.txt'

num_ch = 4
delta_p = 1.0
sigmax2 = 0.05
delta = 12.0
epl = 0.5
rti_T = 0.0
skip_time=16.0
cal_time=16.0

path_start_time = skip_time+cal_time
speed = 1.0 / 2.0

# Initialize rti stuff
# rti_obj = rti.ab_rti(node_loc_fname,num_ch,delta_p,sigmax2,delta,epl,rti_T,skip_time,cal_time)
# rti_obj.set_true_coord_params(path_start_time,speed,pivot_coords_fname,path_ind_fname)

# Initialize moving average rti stuff
# ltb_len = 20
# stb_len = 5
# 
# rti_obj = rti.ma_rti(node_loc_fname,num_ch,delta_p,sigmax2,delta,epl,rti_T,skip_time,cal_time)
# rti_obj.set_mabd_params(ltb_len,stb_len)
# rti_obj.set_true_coord_params(path_start_time,speed,pivot_coords_fname,path_ind_fname)

# Initialize moving average, top M links rti stuff
# ltb_len = 20
# stb_len = 5
# M = 1
# fade_type = 'lse'
#  
# rti_obj = rti.fade_level_rti(node_loc_fname,num_ch,delta_p,sigmax2,delta,epl,rti_T,skip_time,cal_time)
# rti_obj.set_extra_params(ltb_len,stb_len,M,fade_type)
# rti_obj.set_true_coord_params(path_start_time,speed,pivot_coords_fname,path_ind_fname)

# Initialize moving average, top M links rti stuff
ktype = 'epan' # type of kernel
R = np.array(range(-110, 0)) # Range of RSS values
sigma_G2 = 30.
beta_p = 0.99 # short term
beta_q = 0.7 # long term
 
rti_obj = rti.krti(node_loc_fname,num_ch,delta_p,sigmax2,delta,epl,rti_T,skip_time,cal_time)
rti_obj.set_extra_params(R,ktype,sigma_G2,beta_p,beta_q)
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

rti_obj.compute_rmse(true_coord_vec,est_coord_vec,save_type='ps',out_fname=results_fname)
sre,yvals = rti_obj.compute_cdf(true_coord_vec,est_coord_vec)
plt.plot(sre,yvals)
plt.grid()
plt.show()



