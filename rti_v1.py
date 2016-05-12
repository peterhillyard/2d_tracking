import numpy as np
import scipy.spatial.distance as dist
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import time

# In version 2, the score vector is based on the moving average based detection
# by youssef.  We let the score vector be the absolute difference between the 
# short and long term averages.  This way, we do not need a calibration period.
# The model is updated with each added RSS value.



class RTI(object):
    
    #################################
    # Constructor
    #################################
    
    # In this constructor, we save the parameters used in RTI and calculate other
    # parameters we need.
    #
    # Input:
    # sensorCoords : an Nx2 array that stores the (x,y) coordinate of each node
    # mabd         : moving average based detector object
    # delta_p      : the dimension of a pixel
    # sigmax2      : correlation parameter
    # delta        : correlation parameter
    # excessPathLen: width of links
    # calLines     : number of lines included in calibration
    # numCh        : number of channels measured
    # topChs       : number of top channels to add
    # plotSkip     : number of samples to skip before plotting
    # personInAreaThreshold: min image value where we consider someone crossing
    
    # numNodes: number of nodes
    # color_bar_flag: has the color bar been displayed?
    # pixelCoords: (x,y) coordinates of each pixel
    # xVals: x-vals of each pixel
    # yVals: y-vals of each pixel
    # inversion: the inversion matrix
    # prevRSS: vector keeps track of the most recent non-nan rss value
    # sumRSS: keeps a running sum of the RSS during calibration
    # numNonNan: keeps track of the number of non-nan values during calibration for each link
    # maxInds: This says which channels are most reliable
    # calVc : vector of calibration values
    # numLinks: number of link-channels
    # RTIMaxCoord: (x,y) coordinate of RTI image max
    # cur_image: stores the current image
    # calibration_count: keeps track of the number of rss values observed during calibration
    # imageExtent: dimensions of image
    # xValsLen: number of unique x-vals
    # yValsLen: number of unique y-vals
    def __init__(self, image_obj, rss_editor_obj, network, mabd, sigmax2, delta, excessPathLen, personInAreaThreshold):
        self.image_obj = image_obj
        self.rss_obj = rss_editor_obj
        self.network = network
        
        self.delta_p = image_obj.delta_p
        self.sigmax2 = sigmax2
        self.delta = delta
        self.excessPathLen = excessPathLen
        self.personInAreaThreshold = personInAreaThreshold
        self.mabd = mabd
        
        self.inversion = None
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

        self.__initRTI()
        self.__init_image
        
    #################################
    # Methods
    #################################
    
    # This method takes the next rss vector and computes a score vector or adds 
    # in to the calibration period.
    def observe(self,cur_obs,true_loc = np.array([np.nan,np.nan])):
        
        ####################
        # Add rss for calibration period
        self.mabd.observe(cur_obs)
        
        ####################
        # Get the score vector - average of abs diffs on all links
        scoreVec = self.mabd.just_diff
        
        ####################
        # Call RTI and Set the maximum coordinate
        # get image and the coordinate with the max in image
        self.__callRTI(scoreVec)
        
        self.__update_accuracy(true_loc)
        
    # Plots the current image.  This is implemented in a class so that plotting 
    # runs as fast as possible.
    def plot_current_image(self,true_loc=[np.nan,np.nan],pause_time = 0.1):
        
        # Set up the figure if this is the first time through
        if self.is_first_plot:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.current_image, aspect='equal',interpolation='none', origin='lower', extent=self.image_obj.image_extent, vmin=0, vmax = 1)
            self.true_loc_x, = self.ax.plot([],[],'wx',markersize=10,mew=5)
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
        
    # get the accuracy of the points
    def get_accuracy(self):
        print 'Average error of ' + str(np.sqrt(self.current_accuracy/self.num_points_in_accuracy)) + ' ' + self.image_obj.unit
        print 'The min average error is ' + str(np.sqrt(self.current_min_accuracy/self.num_points_in_min_accuracy)) + ' ' + self.image_obj.unit
        return np.sqrt(self.current_accuracy/self.num_points_in_accuracy) 
            
    
    #######################################
    # Helper methods
    #######################################
    def __initRTI(self):

        # Find distances between pixels and transceivers
        # Note that the pixel coordinates includes the [-99,-99] out of network pixel
        DistPixels  = dist.squareform(dist.pdist(self.image_obj.pixel_coords_all[1:]))
        DistPixelAndNode = dist.cdist(self.image_obj.pixel_coords_all[1:], self.network.node_locs_subset)
        DistNodes   = dist.squareform(dist.pdist(self.network.node_locs_subset))
    
        # Find the (inverse of) the Covariance matrix between pixels
        CovPixelsInv       = linalg.inv(self.sigmax2*np.exp(-DistPixels/self.delta))
    
        # Calculate weight matrix for each link.
        
        # linkLocs is a 2d array that stores the tx in the left column, and the rx in the right column
#         linkLocs = self.network.link_ch_database[self.network.master_indexes,:]
#         linkNum = linkLocs.shape[0]
#         W = np.zeros((linkNum, self.image_obj.num_pixels_all-1))
#         for ln in range(linkNum):
#             txNum, rxNum  = linkLocs[ln,1]-1, linkLocs[ln,2]-1
#             print txNum, rxNum
#             ePL           = DistPixelAndNode[:,txNum] + DistPixelAndNode[:,rxNum] - DistNodes[txNum,rxNum]  
#             inEllipseInd  = np.argwhere(ePL < self.excessPathLen)
#             pixelsIn      = len(inEllipseInd)
#             if pixelsIn > 0:
#                 W[ln, inEllipseInd] = 1.0 / float(pixelsIn)
                
        # linkLocs is a 2d array that stores the tx in the left column, and the rx in the right column
        num_nodes = self.network.node_locs_subset.shape[0]
        linkLocs = self.network.link_ch_database[self.network.master_indexes,:]
        linkNum = linkLocs.shape[0]
        W = np.zeros((linkNum, self.image_obj.num_pixels_all-1))
        link_count = 0
        for txNum in range(num_nodes):
            for rxNum in range(num_nodes):
                if txNum == rxNum:
                    continue
                ePL           = DistPixelAndNode[:,txNum] + DistPixelAndNode[:,rxNum] - DistNodes[txNum,rxNum]  
                inEllipseInd  = np.argwhere(ePL < self.excessPathLen)
                pixelsIn      = len(inEllipseInd)
                if pixelsIn > 0:
                    W[link_count, inEllipseInd] = 1.0 / float(pixelsIn)
                link_count += 1
     
        # Compute the projection matrix
        self.inversion       = np.dot(linalg.inv(np.dot(W.T, W) + CovPixelsInv), W.T)
    
    # This computes the current image
    def __callRTI(self, linkMeas):
        
        # perform matrix multiplication
        temp = np.dot(self.inversion, np.array([linkMeas.tolist()]).T)[:,0]
        self.current_image = temp.reshape(self.image_obj.yVals.size, self.image_obj.xVals.size)

        # Get the current location estimate
        if self.current_image.max() > self.personInAreaThreshold:
            self.current_loc_est = self.image_obj.pixel_coords_all[np.argmax(temp)+1,:]
        else:
            self.current_loc_est = np.array([-99.,-99.])   
        
    # update the current accuracy
    def __update_accuracy(self,true_loc):
        if np.logical_not(((true_loc == -99).sum() == 2) | ((self.current_loc_est == -99.).sum() == 2)):
            self.current_accuracy += np.sum((true_loc - self.current_loc_est)**2)
            self.num_points_in_accuracy += 1.
            
        if np.logical_not((true_loc == -99).sum() == 2):
            self.current_min_accuracy += np.min(np.sum((np.tile(true_loc,(self.image_obj.num_pixels_all,1)) - self.image_obj.pixel_coords_all)**2,axis=1))
            self.num_points_in_min_accuracy += 1
    
    # initialize the image
    def __init_image(self):
        
        # Set the first image to be all zeros
        tmp_img = np.zeros(self.image_obj.num_pixels_all)
        self.current_image = np.reshape(tmp_img[1:],(self.image_obj.yVals.size,self.image_obj.xVals.size))
        
        # set the current location to be out of the network
        self.current_loc_est = np.array([np.nan,np.nan])
        
        