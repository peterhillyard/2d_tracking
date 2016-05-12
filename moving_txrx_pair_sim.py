'''
Created on Feb 16, 2016

@author: pete
'''
import numpy as np
import matplotlib.pyplot as plt
import time

class plotting_class:
    
    def __init__(self,):
        self.tx_loc = [0,0] # in meters
        self.rx_loc = [4,0] # in meters
        self.len_max = 8 # in meters
        self.tx_vel = 0.25 # in m/s
        self.rx_vel = 0.25 # in m/s
        
        self.tx_ref_loc = None
        self.rx_ref_loc = None
        
        self.person_loc = [1,1]
        self.person_vel = 1.0 # in m/s
        self.person_rad = 0.4 # in meters
        
        self.node_locs = np.array([np.zeros(80).tolist() + (4*np.ones(80)).tolist(),
                          np.linspace(0,8,80).tolist() + np.linspace(0,8,80).tolist()]).T
        
        self.node_states = np.array([[1,-1,0,1,0,-1,1,0,-1,0],[1,0,-1,0,1,-1,0,1,0,-1]])
#         self.node_states = np.array([[1,-1],[1,-1]])
        self.cur_node_state = 0
        
        self.person_states = np.array([[0,1,0,-1],[1,0,-1,0]])
        self.cur_person_state = 0
        self.sample_period = 0.1 # in secs
        
        # figure stuff
        self.fig = None
        self.ax = None
        self.link_line = None
        self.person_shadow = None
        self.nodes = None
        self.ref_nodes_nonxed = None
        self.ref_nodes_xed = None
        self.is_first_plot = 1
        
    def advance(self,pause_time):
        self.__update_link_pos()
        self.__update_person_pos()
        self.__get_closest_ref()
        self.__is_crossed()
        self.__plot_current_image(pause_time)
    
    # Update the link location
    def __update_link_pos(self):
        tx_tmp = self.tx_loc[1] + self.node_states[0,self.cur_node_state]*self.tx_vel*self.sample_period
        rx_tmp = self.rx_loc[1] + self.node_states[1,self.cur_node_state]*self.rx_vel*self.sample_period
        self.__update_node_state(tx_tmp, rx_tmp)
        self.tx_loc[1] = np.maximum(np.minimum(tx_tmp,self.len_max),0)
        self.rx_loc[1] = np.maximum(np.minimum(rx_tmp,self.len_max),0)
    
    # update the person's location
    def __update_person_pos(self):
        loc_tmp = self.person_loc + self.person_states[:,self.cur_person_state]*self.person_vel*self.sample_period
        self.__update_person_state(loc_tmp)
        self.person_loc[0] = np.maximum(np.minimum(loc_tmp[0],self.rx_loc[0] - 1),1)
        self.person_loc[1] = np.maximum(np.minimum(loc_tmp[1],self.len_max-1),1)
        
    # update the node state
    def __update_node_state(self,tx_tmp,rx_tmp):
        if (tx_tmp > self.len_max) | (rx_tmp > self.len_max) | (tx_tmp < 0) | (rx_tmp < 0):
            self.cur_node_state += 1
            if self.cur_node_state >= self.node_states.shape[1]:
                self.cur_node_state = 0
    
    # update the person state
    def __update_person_state(self,loc_tmp):
        if (loc_tmp[0] < 1) | (loc_tmp[1] < 1) | (loc_tmp[0] > self.rx_loc[0] - 1) | (loc_tmp[1] > self.len_max-1):
            self.cur_person_state += 1
            if self.cur_person_state >= self.person_states.shape[1]:
                self.cur_person_state = 0
                
    # get person shadow datapoints
    def __get_shadow_xpoints(self):
        x_vals = self.person_loc[0] + self.person_rad*np.cos(np.linspace(0,2*3.1415,100))
        return x_vals
        
    def __get_shadow_ypoints(self):
        y_vals = self.person_loc[1] + self.person_rad*np.sin(np.linspace(0,2*3.1415,100))
        return y_vals
    
    def __get_closest_ref(self):
        tx_loc_mat = np.tile(self.tx_loc,(self.node_locs.shape[0],1))
        rx_loc_mat = np.tile(self.rx_loc,(self.node_locs.shape[0],1))
        
        tx_ref_idx = np.argmin(((tx_loc_mat - self.node_locs)**2).sum(axis=1))
        rx_ref_idx = np.argmin(((rx_loc_mat - self.node_locs)**2).sum(axis=1))
        
        self.tx_ref_loc = self.node_locs[tx_ref_idx,:]
        self.rx_ref_loc = self.node_locs[rx_ref_idx,:]
        
    def __is_crossed(self):
        offset = self.tx_loc
        
        my_num = ((np.array(self.person_loc)-np.array(offset))*(np.array(self.rx_loc)-np.array(offset))).sum()
        my_denom = ((np.array(self.rx_loc)-np.array(offset))*(np.array(self.rx_loc)-np.array(offset))).sum()
        my_proj = (my_num/my_denom)*(np.array(self.rx_loc)-np.array(offset))+np.array(offset)
        if np.sqrt(((my_proj - self.person_loc)**2).sum()) < self.person_rad:
            return 1
        else:
            return 0
        
    
    # Plots the current image.  This is implemented in a class so that plotting 
    # runs as fast as possible.
    def __plot_current_image(self,pause_time = 0.1):
        
        # Set up the figure if this is the first time through
        if self.is_first_plot:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-1,5)
            self.ax.set_ylim(-1,9)
            self.ax.set_xlabel('X[m]')
            self.ax.set_ylabel('Y[m]')
            self.ax.set_aspect('equal')
            self.link_line, = self.ax.plot([],[],'ko',lw=2)
            self.person_shadow, = self.ax.plot([],[],'r-',lw=2)
            self.nodes, = self.ax.plot([],[],'rx')
            self.ref_nodes_nonxed, = self.ax.plot([],[],'ko-',lw=2)
            self.ref_nodes_xed, = self.ax.plot([],[],'bo--',lw=2)
            
            self.fig.canvas.draw()
            plt.show(block=False)
            self.is_first_plot = 0
        
         
        # Set current image data and node locations
        self.link_line.set_data([self.tx_loc[0],self.rx_loc[0]],[self.tx_loc[1],self.rx_loc[1]])
        self.nodes.set_data(self.node_locs[:,0],self.node_locs[:,1])
        self.ref_nodes_xed.set_data([self.tx_ref_loc[0],self.rx_ref_loc[0]],[self.tx_ref_loc[1],self.rx_ref_loc[1]])
        self.ref_nodes_nonxed.set_data([self.tx_ref_loc[0],self.rx_ref_loc[0]],[self.tx_ref_loc[1],self.rx_ref_loc[1]])
        xs = self.__get_shadow_xpoints()
        ys = self.__get_shadow_ypoints()
        self.person_shadow.set_data(xs,ys)
        
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.link_line)
        self.ax.draw_artist(self.person_shadow)
        self.ax.draw_artist(self.nodes)      
        if self.__is_crossed(): 
            self.ax.draw_artist(self.ref_nodes_xed) 
        else: 
            self.ax.draw_artist(self.ref_nodes_nonxed);
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        time.sleep(pause_time)
        
my_plot_obj = plotting_class()

while(True):
    my_plot_obj.advance(0.01)



