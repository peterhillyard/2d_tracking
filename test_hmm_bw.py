import numpy as np
import network_class_v1 as aNetwork
import rss_editor_class as rssEditor
import image_class_v1 as anImage
import ghmm_bw as GHMM_BW
import os.path

test_loc_list = ['neals_new_house','gpa_house','neals_old_house']
test_day_list = [['2014_11_10'],
                 ['2016_01_04','2016_01_08','2016_01_15'],
                 ['2012_11_01']]
test_variation_list = [[['']],
                       [[''],['_pete','_pete_bw_tag','_pete_tag'],['_pete_tag','_pete_tag_bw']],
                       [['']]]
test_channel_num_list = [4,4,5]
test_node_mask_list = [[],[4,14,15],[]]

loc_num = 0
day_num = 0
variation_num = 0
test_loc = test_loc_list[loc_num]
test_day = test_day_list[loc_num][day_num]
test_variation = test_variation_list[loc_num][day_num][variation_num]

node_loc_f_name = 'data/node_loc_data/' + test_loc + '/node_loc_' + test_day + '.txt'
# image_data_f_name = 'data/image_data/' + test_loc + '/' + test_day + '/image_data_dp_1p0_lm_0p3.txt'
image_data_f_name = 'data/image_data/' + test_loc + '/' + test_day + '/image_data_dp_0p75_lm_0p2.txt'
rss_data_f_name = 'data/rss_data/' + test_loc + '/rss_' + test_day + test_variation + '_all.txt'
true_loc_f_name = 'data/true_loc_data/' + test_loc + '/true_loc_' + test_day + test_variation + '.txt'

#####################
# Create Network Object
node_locs = np.loadtxt(node_loc_f_name)
num_nodes = node_locs.shape[0]
num_ch = test_channel_num_list[loc_num]
node_list = np.arange(num_nodes)+1
mask = np.ones(node_list.size, dtype=bool)
mask[test_node_mask_list[loc_num]] = False
node_list = node_list[mask]
ch_list = np.arange(num_ch)+1
link_order_choice = 'f'

my_network = aNetwork.aNetwork(node_locs,num_nodes,num_ch,node_list,ch_list,link_order_choice)

#####################
# Create RSS Editor Object
my_rss_editor = rssEditor.RssEditor(my_network)

#####################
# Create Image object
my_image = anImage.anImage(my_network,image_data_f_name)

#####################
# Create gHMM_bw
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 5.0 # shift, 5.0
eta = 4.0 # scale, 4.0
omega = 1.0 # minimum variance, 1.0
my_ghmm_bw = GHMM_BW.GHMM_BW(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, Delta, eta, omega)
my_ghmm_bw.get_num_samples(rss_data_f_name)

######################
# Loop through RSS
if os.path.isfile(true_loc_f_name):
    true_locs = np.loadtxt(true_loc_f_name)
#     true_locs = np.array([[-99.,-99.] for ii in range(2)] + true_locs.tolist())
    true_locs = np.array(true_locs.tolist() + [[-99.,-99.] for ii in range(5)])
else:
    true_locs = np.array([[np.nan,np.nan]])
 
# 
for ii in range(10):
    # Get alpha, beta, gamma
    with open(rss_data_f_name, 'r') as f:
        for line in f:
                 
            my_ghmm_bw.observe(line)
            
    # Run baum welch
    with open(rss_data_f_name, 'r') as f:
        for line in f:
                 
            my_ghmm_bw.observe_bw(line)
                
    my_ghmm_bw.run_imaging(true_locs,0.01)
    my_ghmm_bw.get_accuracy()

        
        
        
        