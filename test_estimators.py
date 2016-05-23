import numpy as np
import network_class_v1 as aNetwork
import rss_editor_class as rssEditor
import image_class_v1 as anImage
import mabd_llcd_class_v1 as anMABD
import hmm_llcd_class_v1 as aHMMllcd
import cce_v1 as CCE
import mle_v1 as MLE
import dnbc_f as DNBC_F
import rti_v1 as RTI
import ghmm_f as GHMM_F
import ghmm_f_w_update as GHMM_FUp
import ghmm_fb as GHMM_FB
import ghmm_v as GHMM_V
import gmle as GMLE
import os.path

# true_loc_f_name = 'data/true_loc_data/neals_old_house/true_loc_2012_11_01.txt'
# true_loc_f_new_name = 'data/true_loc_data/neals_old_house/true_loc_2012_11_01_new.txt'
# true_locs = np.loadtxt(true_loc_f_name)
# out_idx = (true_locs [:,0] == -99.) & (true_locs [:,1] == -99.)
# in_idx = np.logical_not(out_idx)
# true_locs[in_idx,:] *= 3.28084
# np.savetxt(true_loc_f_new_name,true_locs)
# quit()

# This is a list of the locations an measurement campaign was performed
test_loc_list = ['neals_new_house',
                 'gpa_house',
                 'neals_old_house',
                 'pizza_house',
                 'airbnb_atl']

# This is a list of the days the experiments were performed at each location
test_day_list = [['2014_11_10'],
                 ['2016_01_04','2016_01_08','2016_01_15'],
                 ['2012_11_01'],
                 ['2016_05_16'],
                 ['2016_05_17']]

# This is a list of all of the test names run on a given day and location
test_variation_list = [[['']],
                       [[''],['_pete','_pete_bw_tag','_pete_tag'],['_pete_tag','_pete_tag_bw']],
                       [['']],
                       [['_lcom_amal','_lcom_peter','_lcom_amal_walk']],
                       [['_lcom_amal','_lcom_peter','_lcom_qi','_omni_amal','_omni_peter','_omni_qi']]]

# This is a list of the number of channels used at each location
test_channel_num_list = [4,
                         4,
                         5,
                         4,
                         4]

# This is a list of the nodes you want to exclude from the imaging
test_node_mask_list = [[],
                       [4,14,15],
                       [],
                       [],
                       []]

loc_num = 4
day_num = 0
variation_num = 0
test_loc = test_loc_list[loc_num]
test_day = test_day_list[loc_num][day_num]
test_variation = test_variation_list[loc_num][day_num][variation_num]

node_loc_f_name = 'data/node_loc_data/' + test_loc + '/node_loc_' + test_day + '.txt'
image_data_f_name = 'data/image_data/' + test_loc + '/' + test_day + '/image_data_dp_3p0_lm_1p7.txt'
# image_data_f_name = 'data/image_data/' + test_loc + '/' + test_day + '/image_data_dp_0p5_lm_0p2.txt'
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
link_order_choice = 'fb'

my_network = aNetwork.aNetwork(node_locs,num_nodes,num_ch,node_list,ch_list,link_order_choice)

#####################
# Create RSS Editor Object
my_rss_editor = rssEditor.RssEditor(my_network)

#####################
# Create Image object
my_image = anImage.anImage(my_network,image_data_f_name)

#####################
# Create MABD LLCD object
stb_len = 5  # short term buffer length
ltb_len = 20 # long term buffer length
stateb_len = 12 # state buffer length
tau = 0.020   # threshold
num_consec = 2
my_mabd_llcd = anMABD.myMABDClass(my_rss_editor, my_network, stb_len, ltb_len, stateb_len, tau, num_consec)

#################
# HMMC parameters
numStates = 2
 
A = np.zeros((numStates,numStates))
A[0,0] = 0.7
A[0,1:] = (1-A[0,0])/(numStates-1)
A[1:,0] = 0.4
for ii in range(1,int(numStates)):
    A[ii,ii] = 1.0-A[ii,0]
pi = np.zeros((numStates,1))
pi[0,0] = 0.9
pi[1:,0] = (1-pi[0,0])/(numStates-1)
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
  
Delta = 2.0 # shift
eta = 7.0 # scale
omega = 0.75 # minimum variance
 
my_hmmc_1 = aHMMllcd.myHmmBorderClass(A,pi,V,min_p,p127,my_rss_editor,my_network,ltb_len,Delta,eta,omega)

#####################
# Create cce
my_cce = CCE.CCE(my_hmmc_1, my_image, my_rss_editor, my_network)

#####################
# Create mle
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 5.0 # shift
eta = 9.0 # scale
omega = 0.75 # minimum variance
my_mle = MLE.MLE(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, Delta, eta, omega)

#####################
# Create dnbc
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = -2.0 # shift
eta = 9.0 # scale
omega = 1.0 # minimum variance
my_dnbc = DNBC_F.DNBC_F(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, Delta, eta, omega)

#####################
# Create gHMM_f
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 5.0 # shift
eta = 3.0 # scale
omega = 0.75 # minimum variance
my_ghmm_f = GHMM_F.GHMM_F(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, Delta, eta, omega)

#####################
# Create gHMM_fup
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 5.0 # shift
eta = 3.0 # scale
omega = 0.75 # minimum variance
my_ghmm_fup = GHMM_FUp.GHMM_FUp(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, Delta, eta, omega)

#####################
# Create gHMM_fb
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 5.0 # shift
eta = 4.0 # scale
omega = 1.0 # minimum variance
my_ghmm_fb = GHMM_FB.GHMM_FB(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, 30, Delta, eta, omega)

#####################
# Create gHMM_v
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 4.0 # shift
eta = 3.0 # scale
omega = 1.0 # minimum variance
my_ghmm_v = GHMM_V.GHMM_V(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, 20, Delta, eta, omega)

#####################
# Create gMLE
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = -2.0 # shift
eta = 9.0 # scale
omega = 1.0 # minimum variance
my_gmle = GMLE.GMLE(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, Delta, eta, omega)

######################
# Create RTI
sigmax2       = 2.5
delta         = 2.5
excessPathLen = 3.0
personInAreaThreshold = 2.1 #2.1
my_rti = RTI.RTI(my_image,my_rss_editor,my_network,my_mabd_llcd,sigmax2,delta,excessPathLen,personInAreaThreshold)

######################
# Loop through RSS
if os.path.isfile(true_loc_f_name):
    true_locs = np.loadtxt(true_loc_f_name)
#     true_locs = np.array([[-99.,-99.] for ii in range(3)] + true_locs.tolist())
    true_locs = np.array(true_locs[2:,:].tolist() + [[-99.,-99.] for ii in range(2)])
else:
    true_locs = np.array([[np.nan,np.nan]])

# 
counter = 0
with open(rss_data_f_name, 'r') as f:
    for line in f:
            
        my_ghmm_f.observe(line,true_locs[counter,:])
        my_ghmm_f.plot_current_image(true_locs[counter,:],0.01)
        if os.path.isfile(true_loc_f_name):
            counter += 1
        
my_ghmm_f.get_accuracy()
        
        
        
        