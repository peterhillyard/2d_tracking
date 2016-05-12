import numpy as np
import circ_buff_class_v2 as circBuffClass
import sys
import matplotlib.pyplot as plt
import time

# This class implements the maximum likelihood estimator.  It takes a vector of
# RSS values and computes the likelihood of a person being in a pixel given the
# RSS vector.  The pixel with the greatest likelihood is chosen as the
# estimated location.

class MLE:
    # Constructor
    
    # image_obj: an image object
    # rss_obj:  an rss object
    # network:  a network object
    # num_links: the number of links the user wishes to observe
    
    # on_links: a 2d array of the probability of observing an RSS value on a given link when a person is on the link line.  Row r and column c represents the probability of observing RSS(c) on link r
    # off_links: a 2d array of the probability of observing an RSS value on a given link when a person is off the link line.  Row r and column c represents the probability of observing RSS(c) on link r
    # min_p: minimum probability for a given RSS value
    # p127: probability of observing a missed packet
    # off_buff: stores RSS values for each link when no one is in the area.
    
    # is_updated: a flag specifying if the pmfs for each link have been set
    # Delta: the amount of shift in RSS for the on link line case
    # eta: a scalar to multiply the variance of the on link line case
    # omega: the minimum allowable variance for the off link line case
    
    # V_mat: a 2d array that holds the possible RSS values each link can observe
    # num_states: the number of states possible (the number of pixels in the modified image)
    # b_vec: the likelihood of observing an RSS vector for each pixel
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
        self.num_states = self.image_obj.num_pixels_subset
        
        self.b_vec = np.zeros(self.image_obj.num_pixels_subset)
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
        
    # This function takes the current RSS measurement from each link and 
    # converts it to the emission probabilities for each state
    def observe(self,cur_line,true_loc=np.array([np.nan,np.nan])):
        
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
            
            # Get image estimate
            self.__get_image_est()
            
            self.__update_accuracy(true_loc)
            
    # Plots the current image.  This is implemented in a class so that plotting 
    # runs as fast as possible.
    def plot_current_image(self,true_loc=[np.nan,np.nan],pause_time = 0.1):
        
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
        
        # Redraw stuff
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.im)
        self.ax.draw_artist(self.true_loc_x)
        self.ax.draw_artist(self.est_loc_x)
        self.ax.draw_artist(self.node_loc_x)
        
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        time.sleep(pause_time)
        
    # get the accuracy of the points
    def get_accuracy(self):
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
        cur_obs_mat = np.tile(cur_obs,(self.V_mat.shape[1],1)).T
        masked_mat = cur_obs_mat == self.V_mat

        # Extract the probability of the observation on each link for each state
        p_obs_given_off_link = np.sum(self.off_links*masked_mat,axis=1)
        p_obs_given_on_link  = np.sum(self.on_links*masked_mat,axis=1)
        
        p_obs_mat_off = np.tile(p_obs_given_off_link,(self.num_states,1)).T
        p_obs_mat_on  = np.tile(p_obs_given_on_link,(self.num_states,1)).T
        
        # Compute emission probabilities
        tmp1 = self.image_obj.link_pixel_mat_subset*p_obs_mat_on
        tmp2 = np.logical_not(self.image_obj.link_pixel_mat_subset)*p_obs_mat_off
        tmp3 = tmp1 + tmp2
        
        # divide tmp3 into groups of 4.  Multiply and normalize
        prev = np.ones(self.num_states)
        start_mark = 0
        end_mark = 4
        group = end_mark
        while start_mark < self.num_links:
            current = np.product(tmp3[start_mark:np.minimum(self.num_links,end_mark),:],axis=0)
            current = current/np.sum(current)
            prev = (prev*current)/np.sum(prev*current)
            end_mark += group
            start_mark += group
            
        self.b_vec = prev
        
    def __get_image_est(self):
        
        # Get the pixel with the smallest hamming distance        
        self.current_loc_est = self.image_obj.pixel_coords_subset[np.argmax(self.b_vec),:]
        
        # Create the current image
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        tmp_img[self.image_obj.master_pixel_int_idx] = self.b_vec
        self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        
    # update the current accuracy
    def __update_accuracy(self,true_loc):
        if np.logical_not(((true_loc == -99).sum() == 2) | ((self.current_loc_est == -99.).sum() == 2)):
            self.current_accuracy += np.sum((true_loc - self.current_loc_est)**2)
            self.num_points_in_accuracy += 1.
            
        if np.logical_not((true_loc == -99).sum() == 2):
            self.current_min_accuracy += np.min(np.sum((np.tile(true_loc,(self.image_obj.num_pixels_all,1)) - self.image_obj.pixel_coords_all)**2,axis=1))
            self.num_points_in_min_accuracy += 1
        
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

    # initialize the image
    def __init_image(self):
        
        # Set the first image to be all zeros
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        
        # set the current location to be out of the network
        self.current_loc_est = np.array([np.nan,np.nan])
        
        



