import numpy as np
import circ_buff_class_v2 as circBuffClass
import sys

#
# Author: Peter Hillyard
#
# Purpose: This class computes a binary value for if the links in a network are 
# crossed or not.  The user loops through the lines in a .txt rss file, uses the
# observe() function which takes care of all the computations, and calibration, 
# and then the user uses the get_state_est() function to get which links were 
# crossed during that observation.

class myHmmBorderClass(object):
    
    # Input
    # A: Probability of transition in one step.  Matrix with A[i,j] as the
    #    one-step prob of transition from i to j.
    # pi: initial state probability.  Vector pi[i] is the probability of
    #    being in state i at time 0.
    # V: All possible observation values, eg., the alphabet
    # min_p: minimum probability allowed in the for the on and off link probabilities
    # p127: probability of measuring a 127 on a link
    # rss_obj: this object takes care of all the rss manipulation
    # ltb_len: length of the long term buffer for pmf updating
    # Delta: shift in mean between off and on pmf
    # eta: scalar multiplicative constant on variance
    # omega: minimum allowable variance
    
    # Internal
    # num_links: number of links from the rss_obj
    # num_ch: number of channels from the rss_obj
    # on_links: holds the probabilities of observing RSS measurements for each link in the on state
    # off_links: holds the probabilities of observing RSS measurements for each link in the off state
    # off_buff: long term buffer into which we add new RSS vector measurements
    # is_updated: 0 if in calibration period, 1 otherwise
    # V_mat: The list of possible observations repeated by the number of links
    # num_states: The number of states in the Markov model
    # alpha: A circular buffer containing the forward state probabilities for each time index
    # b: the likelihood of measuring each RSS for the on and off state for each link
    # A_col_1: A copy of the first column of the matrix A.  Used to avoid looping
    # A_col_2: A copy of the second column of the matrix A.  Used to avoid looping
    # pi_mat: A tiled version of the pi matrix to avoid looping

    def __init__(self, A, pi, V, min_p, p127, rss_obj, network_obj, ltb_len, Delta, eta, omega):
        self.rss_obj = rss_obj
        self.network = network_obj
        self.num_ch = self.network.num_ch_subset
        self.num_links = self.network.num_links_subset
        
        self.A = A
        self.pi = pi
        self.on_links = np.nan*np.ones((self.num_links,V.shape[0]))
        self.off_links = np.nan*np.ones((self.num_links,V.shape[0]))
        self.min_p = min_p
        self.p127 = p127
        self.off_buff = circBuffClass.myCircBuff(ltb_len,self.num_links)
        self.is_updated = 0
        self.Delta = Delta
        self.eta = eta
        self.omega = omega
        
        self.V_mat = np.tile(V,(self.num_links,1))
        self.num_states = A.shape[0]
        
        self.b = None
        self.A_col_1 = None
        self.A_col_2 = None
        self.pi_mat = None
        self.alpha = None
        
        self.__init_mats()

    #############################################
    #
    #
    #
    #
    # The following methods are to compute the most likely state in the hmm
    #
    #
    #
    #
    #############################################
        
    # This function takes the current RSS measurement from each link and 
    # converts it to the emission probabilities for each state
    def observe(self,cur_line):
        
        # Get the right RSS out
        self.rss_obj.observe(cur_line)
        cur_obs = self.rss_obj.get_rss()
        
        # if we are in calibration time, add the current observation to off 
        # buffer
        if np.logical_not(self.__is_ltb_full()):
            self.__add_obs_to_off_buff(cur_obs)
        
        # if we are done with calibration, and the pmfs have not been set, then
        # set them
        elif np.logical_not(self.is_updated):
            self.__set_static_gaus_pmfs()
            self.is_updated = 1
        
        # if we are done with calibration, and the pmfs are set, then go!
        if self.is_updated:
            
            # Get likelihoods of current vector observation
            self.__update_b_vec(cur_obs)
    
            # make a function call to update alpha
            self.__update_alpha()
            
    # get the most likely state based on the forward algorithm
    #
    def get_state_est(self):
        tmp = self.__get_state_probs()
        
        if np.isnan(tmp).sum() == tmp.size:
            return np.zeros(tmp.shape[1])
        else:
            return tmp[1,:] > tmp[0,:]
        
    # get the current rss time
    def get_time(self):
        return self.rss_obj.get_time()        
    
    ##############################################################
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # Observe helper functions
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ##############################################################
    # Add the current observation to the long term buffer
    def __add_obs_to_off_buff(self,cur_obs):
        self.off_buff.add_observation(cur_obs)
    
    # Check if long term buffer is full
    def __is_ltb_full(self):
        return self.off_buff.is_full()
            
    # Computes the observation given state probability
    def __update_b_vec(self,cur_obs):
        # convert measurement vector into emission probabilities
        # repeat the observation in columns
        cur_obs_mat = np.tile(cur_obs,(self.V_mat.shape[1],1)).T
        masked_mat = cur_obs_mat == self.V_mat

        # Extract the probability of the observation on each link for each state
        p_obs_given_off_link = np.sum(self.off_links*masked_mat,axis=1)
        p_obs_given_on_link  = np.sum(self.on_links*masked_mat,axis=1)
        
        self.b[0,:] = p_obs_given_off_link
        self.b[1,:] = p_obs_given_on_link      
        
    # Compute the forward joint probability alpha.  Compute it for the most 
    #recent observation and add it to alpha's circular buffer
    def __update_alpha(self):
        
        # create the first alpha values when there is only one observation
        if np.isnan(self.alpha).sum() == self.alpha.size:
            alphatmp = self.pi*self.b
            self.alpha = alphatmp/np.tile(alphatmp.sum(axis=0),(2,1))
        
        # create the next alpha values when there is more than one observation
        else:
            row1 = np.sum(self.alpha*self.A_col_1,axis=0)
            row2 = np.sum(self.alpha*self.A_col_2,axis=0)
            alphatmp = np.reshape(np.append(row1,row2),(2,self.num_links))*self.b
            self.alpha = alphatmp/np.tile(alphatmp.sum(axis=0),(2,1))    
        
    # Return the probabilities of being in each state
    def __get_state_probs(self):
        return 1*self.alpha        
    
    ##################################################
    # 
    #
    #
    #
    # The following methods are used to set the pmfs for the links
    #
    #
    #
    #
    ##################################################
    
    # This method defines the on and off pmfs to be static gaussians where the 
    # on pmfs have a lower mean and larger variance
    def __set_static_gaus_pmfs(self):
        if np.logical_not(self.off_buff.is_full()):
            print "The long term buffer is not yet full.  This may give undesirable results"
        
        # median RSS of off-state buffer
        cal_med = self.off_buff.get_no_nan_median()
        
        if (np.sum(cal_med == 127) > 0) | (np.sum(np.isnan(cal_med)) > 0):
            sys.stderr.write('At least one link has a median of 127 or is nan\n\n')
#             quit()
             
        if (np.sum(np.isnan(self.off_buff.get_nanvar())) > 0):
            sys.stderr.write('the long term buffer has a nan')
            quit()
        
        cal_med_mat = np.tile(cal_med,(self.V_mat.shape[1],1)).T
        
        # variance of RSS during calibration
        cal_var = np.maximum(self.off_buff.get_nanvar(),self.omega) #3.0 
        cal_var_mat = np.tile(cal_var,(self.V_mat.shape[1],1)).T
        
        # Compute the off_link emission probabilities for each link
        x = np.exp(- (self.V_mat - cal_med_mat)**2/(2*cal_var_mat/1.0)) # 1.0
        self.off_links = self.__normalize_pmf(x)
        
        # Compute the on_link emission probabilities for each link
        x = np.exp(- (self.V_mat - (cal_med_mat-self.Delta))**2/(self.eta*2*cal_var_mat)) # 3
        self.on_links = self.__normalize_pmf(x)
        
        self.__adjust_for_nans()
    
    # This method takes a matrix where the rows represent an unscaled pmf for a 
    # given link.  It ensures that each pmf retains its shape and scaling values
    # to normalize the pmf to sum to 1 
    def __normalize_pmf(self,x):
        min_p = self.min_p
        p127 = self.p127
                  
        # indexes of where missed packets occur
        zero_idx = False*np.ones(x.shape)
        zero_idx[:,-1] = True
        
        # indexes of where x is less than the minimum allowable probability
        one_idx = x < min_p
        one_idx[:,-1] = False
        
        # indexes where x is above the min prob and not a missed packet
        two_idx = (zero_idx == 0) & (one_idx == 0)
        
        # normalizing parameter
        gamma = (1.0-np.sum(p127*zero_idx + min_p*one_idx,axis=1))/np.sum(x*two_idx,axis=1)
        
        # normalize only the points that are above the min
        x = p127*zero_idx + min_p*one_idx + np.tile(gamma,(self.V_mat.shape[1],1)).T*x*two_idx          
            
        return x
    
    def __adjust_for_nans(self):
        nan_link_idx = np.isnan(np.sum(self.off_links,axis=1))
        
        self.off_links[nan_link_idx,-1] = self.p127
        self.on_links[nan_link_idx,-1] = self.p127
        
        self.off_links[nan_link_idx,:-1] = (1.0-self.p127)/(self.V_mat.shape[1]-1.0)
        self.on_links[nan_link_idx,:-1] = (1.0-self.p127)/(self.V_mat.shape[1]-1.0)
    
    
    ########################################################
    #
    #
    #
    #
    #
    #
    # Methods used in init stage
    #
    #
    #
    #
    #
    #
    ##########################################################
    
    # Initialize matrixes
    def __init_mats(self):
        
        tmp = self.network.link_order_choice
        
        if (tmp == 'a'):
            print "Cannot operate on the a-type format choice. Quitting\n"
            quit()
        
        self.alpha = np.nan*np.ones((2,self.num_links))
        self.b = np.nan*np.ones((2,self.num_links))
        self.off_count = np.zeros(self.num_links)
        self.A_col_1 = np.tile(self.A[:,0],(self.num_links,1)).T
        self.A_col_2 = np.tile(self.A[:,1],(self.num_links,1)).T
        self.pi_mat = np.tile(self.pi[:,0],(self.num_links)).T












