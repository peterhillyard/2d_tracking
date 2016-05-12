
#! /usr/bin/env python

# This script reads packet data from the listen node through the serial port
# and prints one RSS measurement on each line for all tx, rx, ch combination,
# where the rx != tx, both rx and tx are in the list of sensors, and ch is in 
# the list of channels.
#
# In this version, we automate the process of getting the number of nodes and
# the channel list using a packet sniffer function.  You can comment this 
# function call and input these two parameters manually. 
#

# Version History:
#
# Version 1.1:  Initial Release, 2 Oct 2012
# Version 1.2:  

import sys
import serial
import time
import rss as rss
import numpy as np
import network_class_v1 as aNetwork
import rss_editor_class as rssEditor
import image_class_v1 as anImage
import mabd_llcd_class_v1 as anMABD
import mle_v1 as MLE
import ghmm_v as GHMM_V
import rti_v1 as RTI

# Get the number of nodes and channel list automatically
print "Initializing..."
maxNodes, channelList = rss.run_sniffer()
print "\nReady to save data"

# USER: The following serial "file name" changes depending on your operating 
#       system, and what name is assigned to the serial port when your listen 
#       node is plugged in.
serial_filename = rss.serialFileName()
sys.stderr.write('Using USB port file: ' + serial_filename + '\n')
ser = serial.Serial(serial_filename,38400)

# What node numbers are yours, that you want to see output to the file.
# USER:  SET THIS TO THE NODE IDS ASSIGNED TO YOU.  DO NOT INCLUDE THE LISTEN NODE NUMBER
nodeList      = range(1,maxNodes+1)  # 1, ..., 30

# Parameters that are due to our implementation of the listen node.
numNodes      = len(nodeList)
numChs        = len(channelList)
numLinks      = numNodes*(numNodes-1)*numChs
rssIndex      = 3
string_length = maxNodes + 7
suffix        = ['ef','be']  # "0xBEEF"

# Initialize data, output file
nodeSet       = set(nodeList)
channelSet    = set(channelList)
currentLine   = []  # Init serial data buffer "currentLine" as empty.
currentLinkRSS = [127] * numLinks

###########################
# Initialize my imaging method
###########################
node_loc_f_name = 'data/node_loc_data/span_lab/node_loc_2016_02_03.txt'
image_data_f_name = 'data/image_data/span_lab/2016_02_03/image_data_dp_2p0_lm_1p5.txt'
num_ch = numChs

#####################
# Create Network Object
node_locs = np.loadtxt(node_loc_f_name)
num_nodes = node_locs.shape[0]
node_list = np.arange(num_nodes)+1
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
# Create Viterbi HMM
V = np.array(range(-105, -20) + [127]) # possible RSS values
min_p = 0.0001
p127 = 0.03
ltb_len = 30 # length of the long term buffer
Delta = 5.0 # shift
eta = 4.0 # scale
omega = 1.0 # minimum variance
my_ghmm_v = GHMM_V.GHMM_V(my_image, my_rss_editor, my_network, V, min_p, p127, ltb_len, 20, Delta, eta, omega)

######################
# Create RTI
sigmax2       = 0.25
delta         = 3.0
excessPathLen = 0.2
personInAreaThreshold = 1.1 #2.1
my_rti = RTI.RTI(my_image,my_rss_editor,my_network,my_mabd_llcd,sigmax2,delta,excessPathLen,personInAreaThreshold)


# Run forever, adding one integer at a time from the serial port, 
#   whenever an integer is available.
while(1):
    tempInt = ser.read().encode('hex')
    currentLine.append(tempInt)

    # Whenever the end-of-line sequence is read, operate on the "packet" of data.
    if currentLine[-2:] == suffix:
        if len(currentLine) != string_length:
            sys.stderr.write('packet corrupted - wrong string length\n')
            del currentLine[:]
            continue
        currentLineInt = [int(x, 16) for x in currentLine]
        rxId = currentLineInt[2]
        currentCh = currentLineInt[-4]

        if (rxId not in nodeSet) or (currentCh not in channelSet):
            del currentLine[:]
            continue                    
        
        # Each line in the serial data has RSS values for multiple txids.
        # Output one line per txid, rxid, ch combo.
        for txId in nodeList:
            # If the rxId is after the txId, then no problem -- currentCh
            # is also the channel that node txId was transmitting on when
            # node rxId made the measurement, because nodes transmit on a
            # channel in increasing order.
            if rxId > txId: 
                ch = currentCh
            else: 
                ch = rss.prevChannel(channelList, currentCh)
            
            # If the link (tx, rx, ch) is one we are supposed to watch
            if txId != rxId:  
                i = rss.linkNumForTxRxChLists(txId, rxId, ch, nodeList, channelList)
                
                # If the RSS has already been recorded for this link on 
                # this "line", then output the line first, and then restart 
                # with a new line.
                if currentLinkRSS[i] < 127:
                    # Output currentLinkRSS vector
                    cur_line = ' '.join(map(str,currentLinkRSS)) + ' ' + str(time.time()) + '\n'
                    my_rti.observe(cur_line)
                    my_rti.plot_current_image(pause_time=0.0)
                    
                    
                    # Restart with a new line by resetting currentLinkRSS
                    currentLinkRSS = [127] * numLinks
                
                # Store the RSS 
                currentLinkRSS[i] = rss.hex2signedint(currentLine[rssIndex+txId-1])

        # Remove serial data from the buffer.
        currentLine = []
