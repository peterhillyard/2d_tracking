import numpy as np
import circ_buff_class_v2 as circBuffClass

# This class contains all the code necessary to perform Moving Average Based 
# Detection.
#
# After the user creates an MABDClass object, a user uses the observe() function
# to add a current line from a .txt file from listentAllLinks.py.  The observe()
# function takes care of all the computation.  After the user observes, he can 
# use the get_state_est() function, which returns a binary vector of if the links
# were crossed or not.   

class myMABDClass(object):
    
    # Input parameters
    # rss_obj: an rss_editor_class object.
    # stb_len:    length of the short term buffer
    # ltb_len:    length of the long term buffer
    # stateb_len: number of samples in the state buffer
    # tau:        threshold to compare against the relative difference
    # num_consec: the number of alerts needed within the state buff length window to declare a crossing

    # Inside variables
    # num_links: the number of links measured with the rss_obj
    # num_ch: the number of channels used in the rss_editor
    # st_buff:       short term circular buffer
    # lt_buff:       long term circular buffer
    # state_buff:    state circular buffer
    # rel_diff:      the relative difference for each link
    # just_diff:     the difference between the mean of the long and short term buffer
    # output_vec:    the binary vector for each link (0 no crossing, 1 crossing)
    
    def __init__(self, rss_obj, network_obj, stb_len, ltb_len, stateb_len, tau, num_consec):
        self.rss_obj = rss_obj
        self.network = network_obj
        
        self.tau        = 1.0*tau
        self.num_consec = 1.0*num_consec
        
        self.st_buff = circBuffClass.myCircBuff(stb_len,self.network.num_links_subset)
        self.lt_buff = circBuffClass.myCircBuff(ltb_len,self.network.num_links_subset)
        self.state_buff = circBuffClass.myCircBuff(stateb_len,self.network.num_links_subset)

        self.rel_diff = None
        self.just_diff = None
        self.output_vec = None
        
    #######################################
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ########################################
    
    # This method takes the current observation and stores a binary value for each link
    def observe(self,line):
        
        self.rss_obj.observe(line)
        cur_obs = self.rss_obj.get_rss()
        
        # add observation to the buffers
        self.__add_obs_to_buffers(cur_obs)
        
        # Set the binary vector for all measurements
        self.__set_event()
        
    # This method returns the current output vector
    def get_state_est(self):
        return self.output_vec
    
    # This method returns the current time
    def get_time(self):
        return self.rss_obj.get_time()
    
    
    #######################################################################
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ######################################################################## 
    
    # Add the current observation to the buffers
    def __add_obs_to_buffers(self,cur_obs):
        self.st_buff.add_observation(cur_obs)
        self.lt_buff.add_observation(cur_obs)
        
    # Compute the relative difference and add binary vector to state buffer
    def __set_event(self):
        alpha_s = self.st_buff.get_mean()
        alpha_l = self.lt_buff.get_mean()
        
        # The difference between the long and short term mean
        tmp_diff = alpha_l - alpha_s
        tmp_diff[np.isnan(tmp_diff)] = 0
        self.just_diff = tmp_diff
        
        # The relative difference between the long and short term mean
        rd = np.abs((alpha_l - alpha_s)/(1.*alpha_l))
        rd[np.isnan(rd)] = 0
        self.rel_diff = rd
        
        # Add the thresholded binary vector
        self.state_buff.add_observation(self.rel_diff > self.tau)
        
        # Set output binary vector
        tmp = self.state_buff.get_per_row_nansum() >= self.num_consec
        self.output_vec = tmp
        
        