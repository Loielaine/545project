import numpy as np
# train_data=np.loadtxt('train.csv',delimiter=',',skiprows=1,dtype=str)
direction=['W','E','S','N','NE','NW','SW','SE']
direction_num=[i for i in range(8)]
d=dict(zip(direction,direction_num))
city_name=['Atlanta','Philadelphia','Chicago','Boston']
city_num=[0,1,2,3]
city=dict(zip(city_name,city_num))
