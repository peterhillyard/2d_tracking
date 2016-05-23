# This script does the tedious work of creating the codewords for pixels in an
# image.  The image and its parameters are saved to file to be read in quickly
# during testing


import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.spatial.distance as dist

delta_p = 3.0 # check unit type
lambda_val = 1.7 # check unit type
link_to_show = 10

test_loc_list = ['neals_new_house','gpa_house','neals_old_house','span_lab','pizza_house','airbnb_atl']
test_day_list = [['2014_11_10'],
                 ['2016_01_04','2016_01_08','2016_01_15'],
                 ['2012_11_01'],
                 ['2016_02_03'],
                 ['2016_05_16'],
                 ['2016_05_17']]
test_variation_list = [[['']],
                       [[''],['_pete','_pete_bw_tag','_pete_tag'],['_pete_tag','_pete_tag_bw']],
                       [['']],
                       [['']],
                       [['_lcom_amal','_lcom_peter','_lcom_amal_walk']],
                       [['_lcom_amal','_lcom_peter','_lcom_amal_walk']]]
test_unit_list = ['meters','feet','feet','feet','feet','feet']
test_channel_num_list = [4,4,5,4,4,4]

loc_num = 5
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

# Find distances between pixels and transceivers
DistPixels  = dist.squareform(dist.pdist(pixelCoords))
DistPixelAndNode = dist.cdist(pixelCoords, sensorCoords)
DistNodes   = dist.squareform(dist.pdist(sensorCoords))

# Calculate weight matrix for each link.
nodes = len(sensorCoords)
links = nodes*(nodes-1)
pixel_link = np.zeros((link_database.shape[0],pixels),dtype='bool')
for ln in range(link_database.shape[0]):
    txNum, rxNum  = link_database[ln,1], link_database[ln,2]
    ePL           = DistPixelAndNode[:,txNum] + DistPixelAndNode[:,rxNum] - DistNodes[txNum,rxNum]  
    inEllipseInd  = np.argwhere(ePL < lambda_val)
    pixelsIn      = len(inEllipseInd)
    if pixelsIn > 0:
        pixel_link[ln, inEllipseInd] = True





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