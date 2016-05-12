'''
Created on Feb 16, 2016

@author: pete
'''
import numpy as np
import matplotlib.pyplot as plt
import time

class plotting_class:
    
    def __init__(self):
        self.max_h = 8. # in meters, y axis
        self.max_w = 6. # in meters, x axis
        self.sample_period = 0.1 # in secs
        
        self.txrx_1_loc = np.array([[0.,0.],[self.max_w,0.]]) # in meters, tx in row 1, rx in row 2
        self.txrx_2_loc = np.array([[0.,self.max_h],[0.,0.]]) # in meters, tx in row 1, rx in row 2
        self.txrx_1_state = np.array([[1,-1],[1,-1]])
        self.txrx_2_state = np.array([[1,-1],[1,-1]])
        self.cur_txrx_1_state = 0
        self.cur_txrx_2_state = 0
        self.txrx_1_vel = 1.5 # in m/s
        self.txrx_2_vel = 1.5 # in m/s
        
        self.person_loc = np.array([1.,1.])
        self.person_vel = 1.0 # in m/s
        self.person_rad = 0.4 # in meters
        self.person_states = np.array([[0,1,0,-1],[1,0,-1,0]])
#         self.person_states = np.array([[0,0,0,0],[0,0,0,0]])
        self.cur_person_state = 0
        
        self.txrx_1_ref_loc = np.zeros((2,2))
        self.txrx_2_ref_loc = np.zeros((2,2))
        
        self.quad_areas = np.zeros(4)
        self.quad_midpoints = np.zeros((4,2))
        self.cur_loc_est = None
        
        self.ref_node_locs = self.__get_ref_node_locs()
        
        # figure stuff
        self.fig = None
        self.ax = None
        self.link_line_1 = None
        self.link_line_2 = None
        self.person_shadow = None
        self.loc_est = None
        self.ref_nodes_plt = None
        self.txrx_1_nonxed = None
        self.txrx_1_xed = None
        self.txrx_2_nonxed = None
        self.txrx_2_xed = None
        self.is_first_plot = 1
        
    def advance(self,pause_time):
        self.__update_link_pos()
        self.__update_person_pos()
        self.__get_closest_ref()
        self.__get_loc_est()
        self.__plot_current_image(pause_time)
        
    # Update the link location
    def __update_link_pos(self):
        
        # First update first tx-rx pair
        tx1_tmp = self.txrx_1_loc[0,1] + self.txrx_1_state[0,self.cur_txrx_1_state]*self.txrx_1_vel*self.sample_period
        rx1_tmp = self.txrx_1_loc[1,1] + self.txrx_1_state[1,self.cur_txrx_1_state]*self.txrx_1_vel*self.sample_period
        self.__update_txrx_state(tx1_tmp, rx1_tmp,1)
        
        # Next update second tx-rx pair
        tx2_tmp = self.txrx_2_loc[0,0] + self.txrx_2_state[0,self.cur_txrx_2_state]*self.txrx_2_vel*self.sample_period
        rx2_tmp = self.txrx_2_loc[1,0] + self.txrx_2_state[1,self.cur_txrx_2_state]*self.txrx_2_vel*self.sample_period
        self.__update_txrx_state(tx2_tmp, rx2_tmp,2)
        
    # update the person's location
    def __update_person_pos(self):
        loc_tmp = self.person_loc + self.person_states[:,self.cur_person_state]*self.person_vel*self.sample_period
        self.__update_person_state(loc_tmp)
        self.person_loc[0] = np.maximum(np.minimum(loc_tmp[0],self.max_w - 1),1)
        self.person_loc[1] = np.maximum(np.minimum(loc_tmp[1],self.max_h-1),1)
    
    # update the txrx state
    def __update_txrx_state(self,tx_tmp,rx_tmp,pair_num):
        if pair_num == 1:
            if (tx_tmp > self.max_h) | (rx_tmp > self.max_h) | (tx_tmp < 0) | (rx_tmp < 0):
                self.cur_txrx_1_state += 1
                if self.cur_txrx_1_state >= self.txrx_1_state.shape[1]:
                    self.cur_txrx_1_state = 0
                    
            self.txrx_1_loc[0,1] = np.maximum(np.minimum(tx_tmp,self.max_h),0)
            self.txrx_1_loc[1,1] = np.maximum(np.minimum(rx_tmp,self.max_h),0)
        else:
            if (tx_tmp > self.max_w) | (rx_tmp > self.max_w) | (tx_tmp < 0) | (rx_tmp < 0):
                self.cur_txrx_2_state += 1
                if self.cur_txrx_2_state >= self.txrx_2_state.shape[1]:
                    self.cur_txrx_2_state = 0
                    
            self.txrx_2_loc[0,0] = np.maximum(np.minimum(tx_tmp,self.max_w),0)
            self.txrx_2_loc[1,0] = np.maximum(np.minimum(rx_tmp,self.max_w),0)
            
    # update the person state
    def __update_person_state(self,loc_tmp):
        if (loc_tmp[0] < 1) | (loc_tmp[1] < 1) | (loc_tmp[0] > self.max_w - 1) | (loc_tmp[1] > self.max_h-1):
            self.cur_person_state += 1
            if self.cur_person_state >= self.person_states.shape[1]:
                self.cur_person_state = 0
    
    # Get the closest reference node locations
    def __get_closest_ref(self):
        
        # Do first tx-rx pair
        tx1_loc_mat = np.tile(self.txrx_1_loc[0,:],(self.ref_node_locs.shape[0],1))
        rx1_loc_mat = np.tile(self.txrx_1_loc[1,:],(self.ref_node_locs.shape[0],1))
         
        tx1_ref_idx = np.argmin(((tx1_loc_mat - self.ref_node_locs)**2).sum(axis=1))
        rx1_ref_idx = np.argmin(((rx1_loc_mat - self.ref_node_locs)**2).sum(axis=1))
         
        self.txrx_1_ref_loc[0,:] = self.ref_node_locs[tx1_ref_idx,:]
        self.txrx_1_ref_loc[1,:] = self.ref_node_locs[rx1_ref_idx,:]
        
        # Do first tx-rx pair
        tx2_loc_mat = np.tile(self.txrx_2_loc[0,:],(self.ref_node_locs.shape[0],1))
        rx2_loc_mat = np.tile(self.txrx_2_loc[1,:],(self.ref_node_locs.shape[0],1))
         
        tx2_ref_idx = np.argmin(((tx2_loc_mat - self.ref_node_locs)**2).sum(axis=1))
        rx2_ref_idx = np.argmin(((rx2_loc_mat - self.ref_node_locs)**2).sum(axis=1))
         
        self.txrx_2_ref_loc[0,:] = self.ref_node_locs[tx2_ref_idx,:]
        self.txrx_2_ref_loc[1,:] = self.ref_node_locs[rx2_ref_idx,:]
    
    # Determine if the two link lines are crossed
    def __is_crossed(self,pair_num):
        
        if pair_num == 1:
            offset = self.txrx_1_loc[0,:]
            rx_loc = self.txrx_1_loc[1,:]
             
            my_num = ((self.person_loc-offset)*(rx_loc-offset)).sum()
            my_denom = ((rx_loc-offset)*(rx_loc-offset)).sum()
            my_proj = (my_num/my_denom)*(rx_loc-offset)+offset
            if np.sqrt(((my_proj - self.person_loc)**2).sum()) < self.person_rad:
                return 1
            else:
                return 0
        else:
            offset = self.txrx_2_loc[0,:]
            rx_loc = self.txrx_2_loc[1,:]
             
            my_num = ((self.person_loc-offset)*(rx_loc-offset)).sum()
            my_denom = ((rx_loc-offset)*(rx_loc-offset)).sum()
            my_proj = (my_num/my_denom)*(rx_loc-offset)+offset
            if np.sqrt(((my_proj - self.person_loc)**2).sum()) < self.person_rad:
                return 1
            else:
                return 0
    
    # get person shadow datapoints
    def __get_shadow_xpoints(self):
        x_vals = self.person_loc[0] + self.person_rad*np.cos(np.linspace(0,2*3.1415,100))
        return x_vals
         
    def __get_shadow_ypoints(self):
        y_vals = self.person_loc[1] + self.person_rad*np.sin(np.linspace(0,2*3.1415,100))
        return y_vals
    
    def __get_loc_est(self):
        self.__get_areas()
        
        if self.__is_crossed(1) & self.__is_crossed(2):
            self.cur_loc_est = np.array([self.txrx_2_loc[0,0],self.txrx_1_loc[0,1]])
        else:
            self.cur_loc_est = self.quad_midpoints[np.argmax(self.quad_areas),:]
        
    def __get_areas(self):
        self.quad_areas[0] = (self.max_w-self.txrx_2_loc[0,0])*(self.max_h-self.txrx_1_loc[0,1])/(self.max_h*self.max_w)
        self.quad_areas[1] = self.txrx_2_loc[0,0]*(self.max_h-self.txrx_1_loc[0,1])/(self.max_h*self.max_w)
        self.quad_areas[2] = self.txrx_2_loc[0,0]*self.txrx_1_loc[0,1]/(self.max_h*self.max_w)
        self.quad_areas[3] = (self.max_w-self.txrx_2_loc[0,0])*self.txrx_1_loc[0,1]/(self.max_h*self.max_w)
        
        self.quad_midpoints[0,:] = np.array([0.5*(self.max_w-self.txrx_2_loc[0,0])+self.txrx_2_loc[0,0],
                                             0.5*(self.max_h-self.txrx_1_loc[0,1])+self.txrx_1_loc[0,1]])
        self.quad_midpoints[1,:] = np.array([0.5*self.txrx_2_loc[0,0],
                                             0.5*(self.max_h-self.txrx_1_loc[0,1])+self.txrx_1_loc[0,1]])
        self.quad_midpoints[2,:] = np.array([0.5*self.txrx_2_loc[0,0],0.5*self.txrx_1_loc[0,1]])
        self.quad_midpoints[3,:] = np.array([0.5*(self.max_w-self.txrx_2_loc[0,0])+self.txrx_2_loc[0,0],
                                             0.5*self.txrx_1_loc[0,1]])
    
    # This helper function establishes the reference node locations
    def __get_ref_node_locs(self):
        tmp = np.array([np.zeros(self.max_h*10).tolist() + (self.max_w*np.ones(self.max_h*10)).tolist() 
                        + np.linspace(0,self.max_w,self.max_w*10).tolist() + np.linspace(0,self.max_w,self.max_w*10).tolist(),
                        np.linspace(0,self.max_h,self.max_h*10).tolist() + np.linspace(0,self.max_h,self.max_h*10).tolist() 
                        + (self.max_h*np.ones(self.max_w*10)).tolist() + np.zeros(self.max_w*10).tolist()]).T
        return tmp
    
    # Plots the current image.  This is implemented in a class so that plotting 
    # runs as fast as possible.
    def __plot_current_image(self,pause_time = 0.1):
         
        # Set up the figure if this is the first time through
        if self.is_first_plot:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-1,self.max_w+1)
            self.ax.set_ylim(-1,self.max_h+1)
            self.ax.set_xlabel('X[m]')
            self.ax.set_ylabel('Y[m]')
            self.ax.set_aspect('equal')
            self.link_line_1, = self.ax.plot([],[],'ko',lw=2)
            self.link_line_2, = self.ax.plot([],[],'ko',lw=2)
            self.person_shadow, = self.ax.plot([],[],'-',color='gray',lw=2)
            self.ref_nodes_plt, = self.ax.plot([],[],'rx',mew=1)
            self.txrx_1_nonxed, = self.ax.plot([],[],'ko-',lw=2)
            self.txrx_1_xed, = self.ax.plot([],[],'bo--',lw=2)
            self.txrx_2_nonxed, = self.ax.plot([],[],'ko-',lw=2)
            self.txrx_2_xed, = self.ax.plot([],[],'bo--',lw=2)
            self.loc_est, = self.ax.plot([],[],'kx',mew=5)
             
            self.fig.canvas.draw()
            plt.show(block=False)
            self.is_first_plot = 0
         
          
        # Set current image data and node locations
#         self.link_line_1.set_data(self.txrx_1_loc[:,0],self.txrx_1_loc[:,1])
#         self.link_line_2.set_data(self.txrx_2_loc[:,0],self.txrx_2_loc[:,1])
        self.link_line_1.set_data(self.txrx_1_ref_loc[:,0],self.txrx_1_ref_loc[:,1])
        self.link_line_2.set_data(self.txrx_2_ref_loc[:,0],self.txrx_2_ref_loc[:,1])
        
        self.ref_nodes_plt.set_data(self.ref_node_locs[:,0],self.ref_node_locs[:,1])
        self.txrx_1_xed.set_data(self.txrx_1_ref_loc[:,0],self.txrx_1_ref_loc[:,1])
        self.txrx_1_nonxed.set_data(self.txrx_1_ref_loc[:,0],self.txrx_1_ref_loc[:,1])
        self.txrx_2_xed.set_data(self.txrx_2_ref_loc[:,0],self.txrx_2_ref_loc[:,1])
        self.txrx_2_nonxed.set_data(self.txrx_2_ref_loc[:,0],self.txrx_2_ref_loc[:,1])
        self.loc_est.set_data(self.cur_loc_est[0],self.cur_loc_est[1])
        
        
        xs = self.__get_shadow_xpoints()
        ys = self.__get_shadow_ypoints()
        self.person_shadow.set_data(xs,ys)
         
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.ref_nodes_plt)
        self.ax.draw_artist(self.link_line_1)
        self.ax.draw_artist(self.link_line_2)
        self.ax.draw_artist(self.person_shadow)
        self.ax.draw_artist(self.loc_est)
              
        if self.__is_crossed(1): 
            self.ax.draw_artist(self.txrx_1_xed) 
        else: 
            self.ax.draw_artist(self.txrx_1_nonxed)
            
        if self.__is_crossed(2): 
            self.ax.draw_artist(self.txrx_2_xed) 
        else: 
            self.ax.draw_artist(self.txrx_2_nonxed)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        time.sleep(pause_time)
        
my_plot_obj = plotting_class()

while(True):
    my_plot_obj.advance(0.1)



