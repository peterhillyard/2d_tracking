import numpy as np

class anImage:
    # constructor:
    
    # network: a network object that stores the links we desire to observe
    # image_data_f_name: the name of the file that holds the image data.  This file can be created by running create_and_save_image_line.py
    # delta_p: the width and height of a pixel
    # xVals: an array that holds all of the possible x-values each pixel can have
    # yVals: an array that holds all of the possible y-values each pixel can have
    
    # image_extent: a tuple with the min and max x and y value.  used for plotting
    # master_pixel_int_idx: the integer indexes of the pixels that have at least one link passing through
    # num_pixels_all: the total number of pixels in the image (includes "out of network" pixels)
    # num_pixels_subset: the total number of pixels in the modified image (excludes "out of network" pixels)
    # pixel_coords_all: a 2d array that contains all of the pixel coordinates of the image (includes "out of network" pixels)
    # pixel_coords_subset: a 2d array that contains all of the pixel coordinates of the modified image (excludes "out of network" pixels)
    # link_pixel_mat_all: a 2d array of boolean values.  A one in row r and column c means that link r passes through pixel c.  All links are included
    # link_pixel_mat_subset: a 2d array of boolean values.  A one in row r and column c means that link r passes through pixel c.  This matrix differs from the one above in that we exclude links that the user does not want to measure on and we exclude pixels we do not want to get values for.
    
    def __init__(self,my_network, image_data_f_name):
        self.network = my_network
        self.image_data_f_name = image_data_f_name
        
        self.delta_p = None
        self.unit = None
        self.xVals = None
        self.yVals = None
        
        self.image_extent = None
        self.master_pixel_int_idx = None
        self.node_in_pixel_idx = None
        
        self.num_pixels_all = None # includes out of network pixel
        self.num_pixels_subset = None # excludes out of network pixel
        
        self.pixel_coords_all = None # includes out of network pixel
        self.pixel_coords_subset = None # includes out of network pixel
        
        self.link_pixel_mat_all = None # includes out of network pixel
        self.link_pixel_mat_subset = None # includes out of network pixel
        
        self.__get_parameters()
        self.__reduce_parameters()
        self.__find_node_pixels()
        
    # Extract image parameters from file
    def __get_parameters(self):
        
        with open(self.image_data_f_name,'r') as f:
            counter = 0
            for line in f:
                tmp = [i for i in line.split()]
                
                # Get delta_p
                if counter == 0:
                    self.delta_p = float(tmp[2])
                    self.unit = tmp[1]
                    counter += 1
                
                # Get xVals    
                elif counter == 1:
                    self.xVals = np.array([float(x) for x in tmp[1:]])
                    counter += 1
                    
                # Get yVals
                elif counter == 2:
                    self.yVals = np.array([float(x) for x in tmp[1:]])
                    counter += 1
                
                # Get image extent
                elif counter == 3:
                    self.image_extent = tuple([float(x) for x in tmp[1:]])
                    counter += 1
                
                # Get x_pixels
                elif counter == 4:
                    x_pixels = [-99.] + [float(x) for x in tmp[1:]]
                    counter += 1
                
                # Get y_pixels
                elif counter == 5:
                    y_pixels = [-99.] + [float(x) for x in tmp[1:]]
                    self.pixel_coords_all = np.array([x_pixels,y_pixels]).T
                    self.num_pixels_all = self.pixel_coords_all.shape[0]
                    self.link_pixel_mat_all = np.zeros((self.network.num_links_all,self.num_pixels_all),dtype='bool')
                    counter += 1
                
                # Get link_pixel matrix
                elif counter >= 6:
                    self.link_pixel_mat_all[(counter-6),:] = np.array([False] + [bool(int(x)) for x in tmp])
                    counter += 1
        
    
    # Remove pixels that don't have links passing through            
    def __reduce_parameters(self):
        
        # Get the links according to the user preferences in the network object
        tmp_mat = self.link_pixel_mat_all[self.network.master_indexes,:]
        
        # find the pixels where at least one link passes through
        pixel_bin_idx_valid = tmp_mat.sum(axis=0) > 0
        pixel_bin_idx_valid[0] = True # we want to keep the [-99,-99] pixel valid
        self.master_pixel_int_idx = np.arange(self.num_pixels_all)[pixel_bin_idx_valid]
        
        # The matrix subset is the desired links and pixels
        self.link_pixel_mat_subset = tmp_mat[:,self.master_pixel_int_idx]
        
        # Get the new pixel coords and the number of pixels in the subset
        self.pixel_coords_subset = self.pixel_coords_all[self.master_pixel_int_idx,:]
        self.num_pixels_subset = self.pixel_coords_subset.shape[0]
        
    # Indicate which pixels have nodes in them
    def __find_node_pixels(self):
        self.node_in_pixel_idx = np.zeros(self.num_pixels_subset,dtype='bool')
        
        for nn in range(self.network.num_nodes_subset):
            cur_x = self.network.node_locs_subset[nn,0]
            cur_y = self.network.node_locs_subset[nn,1]
            
            in_x_idx = (self.pixel_coords_subset[:,0]-0.5*self.delta_p <= cur_x) & (cur_x <= self.pixel_coords_subset[:,0]+0.5*self.delta_p)
            in_y_idx = (self.pixel_coords_subset[:,1]-0.5*self.delta_p <= cur_y) & (cur_y <= self.pixel_coords_subset[:,1]+0.5*self.delta_p)
            
            self.node_in_pixel_idx = self.node_in_pixel_idx | (in_x_idx & in_y_idx)
        
        
        
    
        
        
        
        
        
        
    


















