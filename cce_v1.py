import numpy as np
import matplotlib.pyplot as plt
import time

# This class implements the closest codeword estimator.  A link line crossing
# detector produces a binary crossing measurement for each link.  This measure-
# ment is then compared to the codeword for each pixel in an image.  The pixel
# with the lowest hamming distance is selected as the estimated location.
#
# The link line crossing detector object must have an "observe" method and a
# "get_state_est" method.  The observe method takes a line from a rss .txt file
# and the get_state_est method outputs the current binary vector.

class CCE:
    
    # Constructor
    
    # llcd_obj - a link line crossing detector obj.  Must have observe and get_state_est methods
    # image_obj - a image object that contains the link-pixel matrix of codewords
    # rss_obj - rss editor object
    # network - a network object
    
    # current_image - A 2d array representing the hamming distance for each pixel
    # current_loc_est - the current location estimate
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
    
    def __init__(self, llcd_obj, image_obj, rss_editor_obj, network):
        self.llcd_obj = llcd_obj
        self.image_obj = image_obj
        self.rss_obj = rss_editor_obj
        self.network = network
        
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
    
    # Observe a new RSS line
    def observe(self,line,true_loc=np.array([np.nan,np.nan])):
        self.llcd_obj.observe(line)
        
        self.__get_image_estimate()
        
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
        
    # get the accuracy of the points
    def get_accuracy(self):
        print 'Average error of ' + str(np.sqrt(self.current_accuracy/self.num_points_in_accuracy)) + ' ' + self.image_obj.unit
        print 'The min average error is ' + str(np.sqrt(self.current_min_accuracy/self.num_points_in_min_accuracy)) + ' ' + self.image_obj.unit
        return np.sqrt(self.current_accuracy/self.num_points_in_accuracy)
        
    ####################################
    # Helper Functions
    ####################################

    # Get the new image
    def __get_image_estimate(self):
        
        # copy binary vector to match the form of the link-pixel matrix
        cur_bin_mat = np.tile(self.llcd_obj.get_state_est(),(self.image_obj.num_pixels_subset,1)).T
        
        # Compute the hamming distance for each pixel
        num_equal_measurements = np.sum(cur_bin_mat != self.image_obj.link_pixel_mat_subset,axis=0)
        
        # Get the pixel with the smallest hamming distance        
        self.current_loc_est = self.image_obj.pixel_coords_subset[np.argmin(num_equal_measurements),:]
        
        # Create the current image
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        tmp_img[self.image_obj.master_pixel_int_idx] = num_equal_measurements
        self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        self.current_image = self.current_image/self.current_image.max()
        
    # update the current accuracy
    def __update_accuracy(self,true_loc):
        if np.logical_not(((true_loc == -99).sum() == 2) | ((self.current_loc_est == -99.).sum() == 2)):
            self.current_accuracy += np.sum((true_loc - self.current_loc_est)**2)
            self.num_points_in_accuracy += 1.
            
        if np.logical_not((true_loc == -99).sum() == 2):
            self.current_min_accuracy += np.min(np.sum((np.tile(true_loc,(self.image_obj.num_pixels_all,1)) - self.image_obj.pixel_coords_all)**2,axis=1))
            self.num_points_in_min_accuracy += 1












        