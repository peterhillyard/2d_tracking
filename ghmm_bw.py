import numpy as np
import circ_buff_class as circBuffClass
import sys
import matplotlib.pyplot as plt
import time

# This class implements a dynamic naive bayesian classifier

class GHMM_BW:
    # Constructor
    
    # image_obj: an image object
    # rss_obj:  an rss object
    # network:  a network object
    # num_links: the number of links the user wishes to observe
    
    # min_p: minimum probability for a given RSS value
    # p127: probability of observing a missed packet
    # off_buff: stores RSS values for each link when no one is in the area.
    
    # is_updated: a flag specifying if the pmfs for each link have been set
    # is_first_observation: a flag used to calculate the alphas correctly
    # Delta: the amount of shift in RSS for the on link line case
    # eta: a scalar to multiply the variance of the on link line case
    # omega: the minimum allowable variance for the off link line case
    
    # V_mat: a 2d array that holds the possible RSS values each link can observe
    # num_states: the number of states possible (the number of pixels in the modified image)
    # b_vec: the likelihood of observing an RSS vector for each pixel
    # alpha:  the forward terms in the forward solution
    # A_mat: the one-step transition probability matrix
    # pi_vec: the initial state probabilities
    # current_image: stores the current image that represents the likelihood at each pixel
    # current_loc_est: the current location estimate (pixel with the greatest likelihood)
    # current_accuracy: current cumulative squared error
    # num_points_in_accuracy: the number of points in accuracy
    # current_min_accuracy: the minimum accuracy achievable with this pixel size
    # num_points_in_min_accuracy: the number of points in the minimum accuracy
    
    # fig - the figure used for plotting
    # ax - the axes for the figure
    # im - the image object
    # true_loc_x - the line object to plot the current true location
    # est_loc_x - the line object to plot the current estimated location
    # node_loc_x - the line object to plot the node locations
    # is_first_plot - a flag that indicates if the figure to plot has already been initialized
    
    def __init__(self,image_obj, rss_editor_obj, network,V,min_p,p127,ltb_len,Delta,eta,omega):
        self.image_obj = image_obj
        self.rss_obj = rss_editor_obj
        self.network = network
        self.num_links = self.network.num_links_subset
        
        self.mew_mat = None
        self.var_mat = None
        self.mew_mat_tmp = None
        self.var_mat_tmp = None
        self.mew_mat_orig = None
        self.var_mat_orig = None
        
        self.min_p = min_p
        self.p127 = p127
        self.off_buff = circBuffClass.myCircBuff(ltb_len,self.num_links)
        self.num_samples = 0
        
        self.is_updated = 0
        self.is_first_obs = 1
        self.is_first_bw_round = 1
        self.cur_sample_idx = -1
        self.Delta = Delta
        self.eta = eta
        self.omega = omega
        
        self.V_mat = np.tile(V,(self.num_links,1))
        self.num_states = self.image_obj.num_pixels_subset
        
        self.b_vec = None #circBuffClass.myCircBuff(self.circ_buff_len,self.num_states)
        self.alpha = None #circBuffClass.myCircBuff(self.circ_buff_len,self.num_states)
        self.beta = None #np.zeros((self.num_states,self.circ_buff_len))
        self.gamma = None #np.zeros((self.num_states,self.circ_buff_len))
        self.A_mat = np.zeros((self.num_states,self.num_states))
        self.pi_vec = np.zeros(self.num_states)
        
        self.current_image = None
        self.current_loc_est = None
        self.current_accuracy = 0
        self.num_points_in_accuracy = 0
        self.current_min_accuracy = 0
        self.num_points_in_min_accuracy = 0
        
        self.fig = None
        self.ax = None
        self.im = None
        self.true_loc_x = None
        self.est_loc_x = None
        self.node_loc_x = None
        self.is_first_plot = 1
        
        self.__init_image()
        self.__init_mats()
        
    # This function takes the current RSS measurement from each link and 
    # converts it to the emission probabilities for each state
    def observe(self,cur_line):
        
        # update the current sample index
        self.cur_sample_idx += 1
        
        # Get the right RSS out
        self.rss_obj.observe(cur_line)
        cur_obs = self.rss_obj.get_nonmiss_rss()
        
        # if we are in calibration time, add the current observation to off 
        # buffer
        if np.logical_not(self.rss_obj.is_no_nonmissedpackets()):
            return
        
        elif np.logical_not(self.__is_ltb_full()):
            self.__add_obs_to_off_buff(cur_obs)
        
        # if we are done with calibration, and the pmfs have not been set, then
        # set them
        elif np.logical_not(self.is_updated):
            if self.is_first_bw_round:
                self.__set_static_gaus_pmfs()
            self.is_updated = 1
        
        # if we are done with calibration, and the pmfs are set, then go!
        if self.is_updated:
            
            # Get likelihoods of current vector observation
            self.__update_b_vec(cur_obs)
            
            # Get forward (alpha) values
            self.__update_alpha()
            
        if self.cur_sample_idx+1 == self.num_samples:
            # Fill in the beta values
            self.__update_beta()
            
            # Fill in the gamma values
            self.__update_gamma()
            
            # reset values
            self.cur_sample_idx = -1
            self.is_first_obs = 1
            self.is_updated = 0
            self.is_first_bw_round = 0
            self.rss_obj.reset()
            self.off_buff.reset_buffer()
            
    # This function takes the current RSS measurement from each link and 
    # runs the baum welch algorithm
    def observe_bw(self,cur_line):
        
        # update the current sample index
        self.cur_sample_idx += 1
        
        # Get the right RSS out
        self.rss_obj.observe(cur_line)
        cur_obs = self.rss_obj.get_nonmiss_rss()
        
        # if we are in calibration time, add the current observation to off 
        # buffer
        if np.logical_not(self.rss_obj.is_no_nonmissedpackets()):
            return
        
        elif np.logical_not(self.__is_ltb_full()):
            self.__add_obs_to_off_buff(cur_obs)
        
        # if we are done with calibration, and the pmfs have not been set, then
        # set them
        elif np.logical_not(self.is_updated):
            self.mew_mat_tmp = np.copy(self.mew_mat)
            self.mew_mat_tmp[:,:] = 0.0
            self.var_mat_tmp = np.copy(self.var_mat)
            self.var_mat_tmp[:,:] = 0.0
            self.is_updated = 1
        
        # if we are done with calibration, and the pmfs are set, then go!
        if self.is_updated:
            
            # Get new HMM parameters
            self.__run_bw(cur_obs)
            
        if self.cur_sample_idx+1 == self.num_samples:
            denom = np.tile(np.nansum(self.gamma,axis=1),(self.num_links,1))
            
            aaa_on = (self.image_obj.link_pixel_mat_subset == 1)*(self.mew_mat_tmp/denom)
            aaa_on[aaa_on == 0.] = np.nan
            aaa_on = np.tile(np.nanmedian(aaa_on,axis=1),(self.image_obj.num_pixels_subset,1)).T
            
            aaa_off = (self.image_obj.link_pixel_mat_subset == 0)*(self.mew_mat_tmp/denom)
            aaa_off[aaa_off == 0.] = np.nan
            aaa_off = np.tile(np.nanmedian(aaa_off,axis=1),(self.image_obj.num_pixels_subset,1)).T
            
            self.mew_mat_tmp = (self.image_obj.link_pixel_mat_subset == 1)*aaa_on + (self.image_obj.link_pixel_mat_subset == 0)*aaa_off
            self.mew_mat_tmp[np.isnan(self.mew_mat_tmp)] = self.mew_mat_orig[np.isnan(self.mew_mat_tmp)]
            
#             self.mew_mat_tmp = self.mew_mat_tmp/denom
#             self.mew_mat_tmp[np.isnan(self.mew_mat_tmp)] = self.mew_mat[np.isnan(self.mew_mat_tmp)]
            
            bbb = np.minimum(np.maximum(self.var_mat_tmp/denom,self.omega),4.0)
            
            aaa_on = (self.image_obj.link_pixel_mat_subset == 1)*bbb
            aaa_on[aaa_on == 0.] = np.nan
            aaa_on = np.tile(np.nanmedian(aaa_on,axis=1),(self.image_obj.num_pixels_subset,1)).T
            
            aaa_off = (self.image_obj.link_pixel_mat_subset == 0)*bbb
            aaa_off[aaa_off == 0.] = np.nan
            aaa_off = np.tile(np.nanmedian(aaa_off,axis=1),(self.image_obj.num_pixels_subset,1)).T
            
            self.var_mat_tmp = (self.image_obj.link_pixel_mat_subset == 1)*aaa_on + (self.image_obj.link_pixel_mat_subset == 0)*aaa_off
            self.var_mat_tmp[np.isnan(self.var_mat_tmp)] = self.var_mat_orig[np.isnan(self.var_mat_tmp)]
            
#             self.var_mat_tmp = np.minimum(np.maximum(self.var_mat_tmp/denom,self.omega),4.0)
#             self.var_mat_tmp[np.isnan(self.var_mat_tmp)] = self.var_mat[np.isnan(self.var_mat_tmp)]
            
            self.mew_mat = np.copy(self.mew_mat_tmp)
            self.var_mat = np.copy(self.var_mat_tmp)
            
            # reset values
            self.cur_sample_idx = -1
            self.is_first_obs = 1
            self.is_updated = 0
            self.is_first_bw_round = 0
            self.current_accuracy = 0.
            self.num_points_in_accuracy = 0
            self.current_min_accuracy = 0.
            self.num_points_in_min_accuracy = 0
            self.rss_obj.reset()
            self.off_buff.reset_buffer()
            
    def get_num_samples(self,fname):
        with open(fname, 'r') as f:
            for line in f:
                self.num_samples += 1
                
        self.b_vec = np.zeros((self.num_states,self.num_samples))
        self.alpha = np.zeros((self.num_states,self.num_samples))
        self.beta = np.zeros((self.num_states,self.num_samples))
        self.beta[:,-1] = 1.0
        self.gamma = np.zeros((self.num_states,self.num_samples))
        
    # Plots the current image.  This is implemented in a class so that plotting 
    # runs as fast as possible.
    def run_imaging(self,true_loc_all=np.array([[np.nan,np.nan]]),pause_time = 0.1):
        
        cur_image_idx = 0
        
        while cur_image_idx < self.num_samples:
        
            self.__get_image_est(cur_image_idx)
            if true_loc_all.shape[0] == 1:
                true_loc = true_loc_all[0,:]
                self.__update_accuracy(true_loc)
            else:
                true_loc = true_loc_all[cur_image_idx,:]
                self.__update_accuracy(true_loc)
            
            
            # Set up the figure if this is the first time through
            if self.is_first_plot:
                self.fig, self.ax = plt.subplots()
                self.im = self.ax.imshow(self.current_image, aspect='equal',interpolation='none', origin='lower', extent=self.image_obj.image_extent, vmin=0, vmax = 1)
                self.true_loc_x, = self.ax.plot([],[],'wx',markersize=15,mew=5)
                self.est_loc_x, = self.ax.plot([],[],'wo',markersize=15)
                self.node_loc_x, = self.ax.plot([],[],'kd')
                
                self.fig.canvas.draw()
                plt.show(block=False)
                self.is_first_plot = 0
            
             
            # Set current image data and node locations
            self.im.set_array(self.current_image)
            self.node_loc_x.set_data(self.network.node_locs_all[:,0],self.network.node_locs_all[:,1])
            
            # True location X
            if (true_loc[0] != -99.) & (true_loc[1] != -99.):
                self.true_loc_x.set_data(true_loc[0],true_loc[1])
            else:
                self.true_loc_x.set_data(np.nan,np.nan)
            
            # Estimate location O
            if (self.current_loc_est[0] != -99.) & (self.current_loc_est[1] != -99.):
                self.est_loc_x.set_data(self.current_loc_est[0],self.current_loc_est[1])
            else:
                self.est_loc_x.set_data(np.nan,np.nan)
            
            self.ax.draw_artist(self.ax.patch)
            
            self.ax.draw_artist(self.im)
            self.ax.draw_artist(self.true_loc_x)
            self.ax.draw_artist(self.est_loc_x)
            self.ax.draw_artist(self.node_loc_x)
            
            self.fig.canvas.update()
            self.fig.canvas.flush_events()
            time.sleep(pause_time)
            
            cur_image_idx += 1
        
    # get the accuracy of the points
    def get_accuracy(self):
        if self.num_points_in_accuracy == 0:
            print "No accuracy to report. A person was never detected."
            return -1
        else:
            print 'Average error of ' + str(np.sqrt(self.current_accuracy/self.num_points_in_accuracy)) + ' ' + self.image_obj.unit
            print 'The min average error is ' + str(np.sqrt(self.current_min_accuracy/self.num_points_in_min_accuracy)) + ' ' + self.image_obj.unit
            return np.sqrt(self.current_accuracy/self.num_points_in_accuracy)                          
            
    #-----------------------------
    # Helper functions
    #-----------------------------    
            
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
        cur_obs_mat = np.tile(cur_obs,(self.num_states,1)).T

        tmp = -0.5*np.sum(((cur_obs_mat - self.mew_mat)**2)/self.var_mat,axis=0)
        tmp -= tmp.max()
        tmp = np.exp(tmp)
        
        if tmp.sum() != 0:
            tmp = tmp/tmp.sum()
            
        self.b_vec[:,self.cur_sample_idx] = tmp
        
    # Compute the forward joint probability alpha.  Compute it for the most 
    #recent observation and add it to alpha's circular buffer
    def __update_alpha(self):
        # create the first alpha values when there is only one observation
        if self.is_first_obs:
            alphatmp = self.pi_vec*self.b_vec[:,self.cur_sample_idx]
            alphatmp = alphatmp/np.sum(alphatmp)
            self.alpha[:,self.cur_sample_idx] = alphatmp
            self.is_first_obs = 0
        
        # create the next alpha values when there is more than one observation
        else:
            alphatmp = np.dot(self.alpha[:,self.cur_sample_idx-1],self.A_mat)*self.b_vec[:,self.cur_sample_idx]
            self.alpha[:,self.cur_sample_idx] = alphatmp/alphatmp.sum()
            
    # Compute the backward (beta) values
    def __update_beta(self): 
        cur_beta_idx = self.num_samples-2
        
        while cur_beta_idx >= 0:
            betatmp = np.dot(self.A_mat,self.b_vec[:,cur_beta_idx+1]*self.beta[:,cur_beta_idx+1])
            self.beta[:,cur_beta_idx] = betatmp/betatmp.sum()
            cur_beta_idx -= 1
        
    # Compute the gamma values
    def __update_gamma(self):
        numer = self.alpha*self.beta
        self.gamma = numer/np.tile(numer.sum(axis=0),(self.num_states,1))
        
    # Run Baum Welch Algorithm
    def __run_bw(self,cur_obs):
        
        if np.isnan(self.gamma[:,self.cur_sample_idx]).sum() == 0:
            gamma_mat = np.tile(self.gamma[:,self.cur_sample_idx],(self.num_links,1))
            
            obs_mat = np.tile(cur_obs,(self.num_states,1)).T
            
            
            self.mew_mat_tmp += obs_mat*gamma_mat
            self.var_mat_tmp += ((obs_mat-self.mew_mat)**2)*gamma_mat
            
    
    # Update the current image and location estimate
    def __get_image_est(self,cur_image_idx):
        
        if np.isnan(self.gamma[:,cur_image_idx]).sum() == 0:
            # Get the pixel with the smallest hamming distance        
            self.current_loc_est = self.image_obj.pixel_coords_subset[np.argmax(self.gamma[:,cur_image_idx]),:]
            
            # Create the current image
            tmp_img = np.zeros(self.image_obj.num_pixels_all)
            tmp_img[self.image_obj.master_pixel_int_idx] = self.gamma[:,cur_image_idx]
            self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        else:
            # Get the pixel with the smallest hamming distance        
            self.current_loc_est = np.array([-99.,-99.])
            
            # Create the current image
            tmp_img = np.zeros(self.image_obj.num_pixels_all)
            self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
    
    # update the current accuracy
    def __update_accuracy(self,true_loc):
        if np.logical_not(((true_loc == -99.).sum() == 2) | ((self.current_loc_est == -99.).sum() == 2)):
            self.current_accuracy += np.sum((true_loc - self.current_loc_est)**2)
            self.num_points_in_accuracy += 1.
            
        if np.logical_not((true_loc == -99.).sum() == 2):
            self.current_min_accuracy += np.min(np.sum((np.tile(true_loc,(self.image_obj.num_pixels_all,1)) - self.image_obj.pixel_coords_all)**2,axis=1))
            self.num_points_in_min_accuracy += 1
        
    # This method defines the on and off pmfs to be static gaussians where the 
    # on pmfs have a lower mean and larger variance
    def __set_static_gaus_pmfs(self):
        
        mew_off = self.off_buff.get_no_nan_median()
        mew_on = mew_off - self.Delta
        
        var_off = np.maximum(self.off_buff.get_nanvar(),self.omega)
        var_on = self.eta*var_off
        
        self.mew_mat = np.tile(mew_off,(self.num_states,1)).T*(self.image_obj.link_pixel_mat_subset == 0)
        self.mew_mat += np.tile(mew_on,(self.num_states,1)).T*self.image_obj.link_pixel_mat_subset
        
        self.mew_mat_orig = np.copy(self.mew_mat)
        
        self.var_mat = np.tile(var_off,(self.num_states,1)).T*(self.image_obj.link_pixel_mat_subset == 0)
        self.var_mat += np.tile(var_on,(self.num_states,1)).T*self.image_obj.link_pixel_mat_subset
        
        self.var_mat_orig = np.copy(self.var_mat) 

    # initialize the image
    def __init_image(self):
        
        # Set the first image to be all zeros
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        
        # set the current location to be out of the network
        self.current_loc_est = np.array([np.nan,np.nan])
        
    # Plot the mean RSS for a given link
    def plot_avg_image(self,link_avg,link_num):
        
        link_loc_info = self.network.link_ch_database[link_num,:]
        tx_loc = self.network.node_locs_subset[link_loc_info[1]-1,:]
        rx_loc = self.network.node_locs_subset[link_loc_info[2]-1,:]
        
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        tmp_img[self.image_obj.master_pixel_int_idx] = link_avg[link_num,:]
        tmp_img = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        
        plt.figure(1)
        plt.imshow(tmp_img, aspect='equal',interpolation='none', origin='lower', extent=self.image_obj.image_extent, vmin=link_avg[link_num,:].min(), vmax = link_avg[link_num,:].max())
        plt.plot([tx_loc[0],rx_loc[0]],[tx_loc[1],rx_loc[1]],'ko-')
        plt.show(block=False)
        
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        tmp_img[self.image_obj.master_pixel_int_idx] = self.mew_mat_orig[link_num,:]
        tmp_img = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        
        plt.figure(2)
        plt.imshow(tmp_img, aspect='equal',interpolation='none', origin='lower', extent=self.image_obj.image_extent, vmin=link_avg[link_num,:].min(), vmax = link_avg[link_num,:].max())
        plt.plot([tx_loc[0],rx_loc[0]],[tx_loc[1],rx_loc[1]],'ko-')
        plt.show(block=True)
        
        
    
    # set up pi vector and A matrix
    def __init_mats(self):
        
        min_trans_prob = 0.0001
        
        # Get the indexes for the border pixels
        minx_idx = self.image_obj.pixel_coords_all[:,0] == np.min(self.image_obj.xVals)
        maxx_idx = self.image_obj.pixel_coords_all[:,0] == np.max(self.image_obj.xVals)
        miny_idx = self.image_obj.pixel_coords_all[:,1] == np.min(self.image_obj.yVals)
        maxy_idx = self.image_obj.pixel_coords_all[:,1] == np.max(self.image_obj.yVals)
        border_pixels_idx_all = minx_idx | maxx_idx | miny_idx | maxy_idx
        border_pixels_idx_subset = border_pixels_idx_all[self.image_obj.master_pixel_int_idx]
        num_border_pixels = np.sum(border_pixels_idx_subset)
        
        
        # Set pi vector values
        self.pi_vec[0] = 0.9
        self.pi_vec[1:] = (1.0-self.pi_vec[0])/(self.pi_vec.size-1.)
        
        # Set A matrix values
        self.A_mat[0,0] = 0.75
        self.A_mat[0,border_pixels_idx_subset] = (1.0 - self.A_mat[0,0])/num_border_pixels
        
        dp = self.image_obj.delta_p
        shifts = np.array([[0,0],[-dp,0],[-dp,-dp],[0,-dp],[dp,-dp],[dp,0],[dp,dp],[0,dp],[-dp,dp]])
        for pp in range(1,self.num_states):
            
            # is it a border pixel
#             
                # Get pixels that are neighbors
            cur_pixel = self.image_obj.pixel_coords_subset[pp,:]
            
            neighbor_idx = np.zeros(self.num_states,dtype='bool')
            for ss in shifts:
                neighbor = cur_pixel + ss
                
                tmp1 = (self.image_obj.pixel_coords_subset[:,0] == neighbor[0])
                neighbor_idx = neighbor_idx | (tmp1 & (self.image_obj.pixel_coords_subset[:,1] == neighbor[1]))
            
            if border_pixels_idx_subset[pp]:
                neighbor_idx[0] = True
            
            self.A_mat[pp,neighbor_idx == 0] = min_trans_prob
            self.A_mat[pp,neighbor_idx] = (1. - (self.num_states-neighbor_idx.sum())*min_trans_prob)/neighbor_idx.sum()
    
        stuff = 1
        