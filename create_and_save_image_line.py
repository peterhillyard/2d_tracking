# This script does the tedious work of creating the codewords for pixels in an
# image.  The image and its parameters are saved to file to be read in quickly
# during testing


# import numpy as np
# import matplotlib.pyplot as plt
# import os.path
# 
# delta_p = 2.0 # in feet
# dp_unit = 'meters'
# link_to_show = 700
# dp_str = [str(int(delta_p)),str(delta_p-int(delta_p))[2:]]
# 
# f_out_name = 'data/image_data/neals_new_house/2014_11_10/image_data_dp_' + dp_str[0] + 'p' + dp_str[1] + '.txt'
# # if os.path.isfile(f_out_name):
# #     print "This file already exists!"
# #     quit()
# 
# sensorCoords = np.loadtxt('data/node_loc_data/neals_new_house/node_loc_2014_11_10.txt')
# numNodes = sensorCoords.shape[0]
# numCh = 4
# 
# # Get link database
# counter = 0
# link_database = []
# for tx in range(numNodes):
#     for rx in range(numNodes):
#         if tx != rx:
#             link_database.append([counter,tx,rx])
#             counter += 1
# link_database = np.array(link_database)

import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.spatial.distance as dist

delta_p = 1.0 # check unit type
lambda_val = 0.0 # check unit type
link_to_show = 715

test_loc_list = ['neals_new_house','gpa_house','neals_old_house']
test_day_list = [['2014_11_10'],
                 ['2016_01_04','2016_01_08','2016_01_15'],
                 ['2012_11_01']]
test_variation_list = [[['']],
                       [[''],['_pete','_pete_bw_tag','_pete_tag'],['_pete_tag','_pete_tag_bw']],
                       [['']]]
test_unit_list = ['meters','feet','feet']
test_channel_num_list = [4,4,5]

loc_num = 2
day_num = 0
variation_num = 0
test_loc = test_loc_list[loc_num]
test_day = test_day_list[loc_num][day_num]
test_variation = test_variation_list[loc_num][day_num][variation_num]
dp_unit = test_unit_list[loc_num]
numCh = test_channel_num_list[loc_num]

dp_str = [str(int(delta_p)),str(delta_p-int(delta_p))[2:]]
lm_str = [str(int(lambda_val)),str(lambda_val-int(lambda_val))[2:]]


f_out_name = 'data/image_data/' + test_loc + '/' + test_day + '/image_data_dp_' + dp_str[0] + 'p' + dp_str[1] + '_lm_' + lm_str[0] + 'p' + lm_str[1] + '.txt'
# if os.path.isfile(f_out_name):
#     print "This file already exists!"
#     quit()

sensorCoords = np.loadtxt('data/node_loc_data/' + test_loc + '/node_loc_' + test_day + '.txt')
numNodes = sensorCoords.shape[0]


# Get link database
counter = 0
link_database = []
for tx in range(numNodes):
    for rx in range(numNodes):
        if tx != rx:
            link_database.append([counter,tx,rx])
            counter += 1
link_database = np.array(link_database)
            



######################
# Pixels!
print "Creating pixel-link database..."
personLL        = sensorCoords.min(axis=0)
personUR        = sensorCoords.max(axis=0)
if personLL[0] == personUR[0]:
    xVals = np.array([0])-delta_p/2.
else:
    xVals  = np.arange(personLL[0], personUR[0]+1+delta_p/2., delta_p)-delta_p/2.

if personLL[1] == personUR[1]:
    yVals = np.array([0]) -delta_p/2.
else:
    yVals  = np.arange(personLL[1], personUR[1]+1+delta_p/2., delta_p)-delta_p/2.

cols   = len(xVals)
pixels = cols * len(yVals)  # len(yVals) is the number of rows of pixels
imageExtent   = (min(xVals) - delta_p/2, max(xVals) + delta_p/2, 
                 min(yVals) - delta_p/2, max(yVals) + delta_p/2)

# fill the first row, then the 2nd row, etc.
pixelCoords = np.array([[xVals[i%cols], yVals[i/cols]] for i in range(pixels)])
pixelCoords_left = np.array([(pixelCoords[:,0] - delta_p/2.0).tolist(),pixelCoords[:,1].tolist()]).T
pixelCoords_right = np.array([(pixelCoords[:,0] + delta_p/2.0).tolist(),pixelCoords[:,1].tolist()]).T
pixelCoords_up = np.array([pixelCoords[:,0].tolist(),(pixelCoords[:,1] + delta_p/2.0).tolist()]).T
pixelCoords_down = np.array([pixelCoords[:,0].tolist(),(pixelCoords[:,1] - delta_p/2.0).tolist()]).T

pixel_link = np.zeros((link_database.shape[0],pixels),dtype='bool')

for tx in range(numNodes):
    for rx in range(numNodes):
        if tx == rx:
            continue
        
        ll_1 = link_database[(link_database[:,1] == tx) & (link_database[:,2] == rx),0][0]
        ll_2 = link_database[(link_database[:,1] == rx) & (link_database[:,2] == tx),0][0]

#         if ll_1 == 7
        
        p1 = sensorCoords[tx,:]
        p2 = sensorCoords[rx,:]
        
        minx = np.minimum(p1[0], p2[0])
        maxx = np.maximum(p1[0], p2[0])
    
        miny = np.minimum(p1[1], p2[1])
        maxy = np.maximum(p1[1], p2[1])

        for pp in range(pixels):
            v = pixelCoords[pp,:]-p1
            s = p2 - p1
            
            p_proj = np.dot(v,s)/np.dot(s,s)*s + p1
            
            p_c = pixelCoords[pp,:]    
    
            # Are the nodes in the current pixel or are any of the nodes in the same location as the pixel
            in1x = (p_c[0] - delta_p/2. <= p1[0]) & (p1[0] <= p_c[0] + delta_p/2.)
            in1y = (p_c[1] - delta_p/2. <= p1[1]) & (p1[1] <= p_c[1] + delta_p/2.)
            in2x = (p_c[0] - delta_p/2. <= p2[0]) & (p2[0] <= p_c[0] + delta_p/2.)
            in2y = (p_c[1] - delta_p/2. <= p2[1]) & (p2[1] <= p_c[1] + delta_p/2.)
            ep1 = (p_c[0] == p1[0]) & (p_c[1] == p1[1])
            ep2 = (p_c[0] == p2[0]) & (p_c[1] == p2[1])
        
            isValid = ((p_proj[0] >= minx) & (p_proj[0] <= maxx)) & ((p_proj[1] >= miny) & (p_proj[1] <= maxy))
#             isValid = isValid | ((in1x&in1y) | (in2x&in2y))
            
            if (np.sum((p_proj - p_c)**2) == 0) | (ep1 | ep2):
                isValid = 0
#                 pixel_link[ll_1,pp] = True
#                 pixel_link[ll_2,pp] = True
            
            if isValid:
            
                p_c = pixelCoords[pp,:]
                a = p_proj - p_c
                b1 = pixelCoords_left[pp,:] - p_c
                b2 = pixelCoords_right[pp,:] - p_c
                b3 = pixelCoords_up[pp,:] - p_c
                b4 = pixelCoords_down[pp,:] - p_c
                
                theta = [np.arccos(np.dot(a,b1)/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b1**2))))*180/np.pi]
                theta.append(np.arccos(np.dot(a,b2)/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b2**2))))*180/np.pi)
                theta.append(np.arccos(np.dot(a,b3)/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b3**2))))*180/np.pi)
                theta.append(np.arccos(np.dot(a,b4)/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b4**2))))*180/np.pi)
                h = np.min(np.abs(delta_p/(2.*np.cos(np.array(theta)))))
                h_p = np.sqrt(np.sum((p_proj - p_c)**2))
                
                
                pixel_link[ll_1,pp] = (h_p<=h)
                pixel_link[ll_2,pp] = (h_p<=h)

pixel_link = np.tile(pixel_link,(numCh,1))


#############################
# Save important things to file
print "Saving to file..."
with open(f_out_name,'w') as f:
    f.write('delta_p, ' + dp_unit + ', ' + str(delta_p) + '\n')
    
    f.write('xVals, ')
    for item in xVals.tolist():
        f.write("%s " % item)
    f.write('\n')
    
    f.write('yVals, ')
    for item in yVals.tolist():
        f.write("%s " % item)
    f.write('\n')
    
    f.write('Image_extent, ')
    for item in imageExtent:
        f.write("%s " % item)
    f.write('\n')
    
    f.write('x_pixel, ')
    for item in pixelCoords[:,0].tolist():
        f.write("%s " % item)
    f.write('\n')
    
    f.write('y_pixel, ')
    for item in pixelCoords[:,1].tolist():
        f.write("%s " % item)
    f.write('\n')
    
    for ll in range(pixel_link.shape[0]):
        for item in pixel_link[ll,:].astype(int).tolist():
            f.write("%s " % item)
        f.write('\n')

#############################
# Plot stuff

# Plot an image of the number of links that cross each pixel
# plt.imshow(np.reshape(np.sum(pixel_link,axis=0),(yVals.size,xVals.size)), interpolation='none', origin='lower', extent=imageExtent)
# plt.show()
# plt.close()
 
# plot the sensor locations and the pixel locations
plt.plot(sensorCoords[:,0],sensorCoords[:,1],'ro')
plt.plot(pixelCoords[:,0],pixelCoords[:,1],'bx')

# Plot the desired link line 
tx = link_database[link_to_show,1]
rx = link_database[link_to_show,2]
 
p1 = sensorCoords[tx,:]
p2 = sensorCoords[rx,:]

# plot the pixels that are crossed by the desired link line
for pp in range(pixel_link.shape[1]):
    if pixel_link[link_to_show,pp] == 1:
        plt.plot(pixelCoords[pp,0],pixelCoords[pp,1],'ko')
        
# plot the link line
plt.plot([p1[0],p2[0]],[p1[1],p2[1]],'r-',lw=2)

# Draw pixel boundaries
min_y = np.min(pixelCoords[:,1]) - delta_p/2.
max_y = np.max(pixelCoords[:,1]) + delta_p/2.
min_x = np.min(pixelCoords[:,0]) - delta_p/2.
max_x = np.max(pixelCoords[:,0]) + delta_p/2.
for yy in range(yVals.size+1):
    plt.plot([min_x,max_x],[min_y+yy*delta_p,min_y+yy*delta_p],'k-')
for xx in range(xVals.size+1):
    plt.plot([min_x+xx*delta_p,min_x+xx*delta_p],[min_y,max_y],'k-')
     
 
 
plt.plot()
plt.axis('equal')
plt.show()