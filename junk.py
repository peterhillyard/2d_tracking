import numpy as np
import matplotlib.pyplot as plt

N = 10000
tmp1 = 0.35*(np.sqrt(300.)*np.random.randn(N)+1500.)
tmp2 = 0.4*(np.sqrt(100.)*np.random.randn(N)+500.)
tmp3 = 0.25*(np.sqrt(100.)*np.random.randn(N)+200.)

all_vals = tmp1+tmp2+tmp3

y, x = np.histogram(all_vals)
nbins = y.size
plt.hist(all_vals, bins=nbins, alpha=0.5)
plt.show()

quit()



import numpy as np

num_off = 50
num_on = 100

p_on = 0.9
p_off = 0.99999999999

rss_vals = np.array((-60*np.ones(num_off)).tolist() + (-64*np.ones(num_on)).tolist() + (-60*np.ones(num_off)).tolist())
gamma_off = np.array((p_off*np.ones(num_off)).tolist() + (p_on*np.ones(num_on)).tolist() + (p_off*np.ones(num_off)).tolist())
gamma_on = 1-gamma_off

mew_off = (rss_vals*gamma_off).sum()/gamma_off.sum()
mew_on = (rss_vals*gamma_on).sum()/gamma_on.sum()

print mew_off
print mew_on
        





quit()
# import rti
import numpy as np

startPathTime = 40000
speed = 1./4000

pathInd = np.loadtxt('data/neal_new_house_path.txt')
pivotCoords = np.loadtxt('data/neal_new_house_pivot_coords.txt')
rss_data_f_name = 'data/rss_data/neals_new_house/rss_2014_11_10_all.txt'

true_coord = []

with open(rss_data_f_name, 'r') as f:
    for line in f:

        lineList    = [int(i) for i in line.split()]
        time_ms     = lineList.pop(-1)  # remove last element
        
        actualCoord = rti.calcActualPosition(time_ms, pivotCoords, pathInd, startPathTime, speed)
        
        if len(actualCoord) == 0:
            true_coord.append([-99.,-99.])
        else:
            true_coord.append(actualCoord.tolist())
        
np.savetxt('data/true_loc_data/neals_new_house/true_loc_2014_11_10.txt',np.array(true_coord))