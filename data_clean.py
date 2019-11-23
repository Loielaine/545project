import numpy as np
import re
# train_data=np.loadtxt('train.csv',delimiter=',',skiprows=1,dtype=str)
# direction=['W','E','S','N','NE','NW','SW','SE']
# direction_num=[i for i in range(8)]
# d=dict(zip(direction,direction_num))
# city_name=['Atlanta','Philadelphia','Chicago','Boston']
# city_num=[0,1,2,3]
# city=dict(zip(city_name,city_num))
import pandas as pd
data1 = pd.read_csv("weather1.csv")
data1=data1.values
delete_idx=[]
month=[]
day=[]
hours=[]
useful_idx=[]
def del0(x):
    if x[0]=='0':
        x=x[1]
    return x
for i in range(data1[:,1].size):
    date=data1[i,1].split('-')
    if date[0]== '2019' and date[1]=='01':
        pass
    elif date[1] in ['06','07','08','09','10','11','12']:
        pass
    else:
        delete_idx.append(i)
del2= np.delete(data1,delete_idx,0)
del_idx=[]
for i in range(del2[:,1].size):
    if '94846' in str(del2[i,0]):
        del2[i,0]= 1
    elif '14739' in str(del2[i,0]):
        del2[i,0]= 2
    elif '03888' in str(del2[i,0]):
        del2[i,0]=3
    else:
        del_idx.append(i)
    date=del2[i,1].split('-')
    # hour=re.findall('(?<=T)[^:]*(?=:)',date[2])[0]
    if date[2].split(':')[1]=='59':
        useful_idx.append(i)
        continue
    # else:
    #     month.append(int(del0(date[1])))
    #     day.append(del0(date[2].split('T')))
    #     hours.append(hour)
# print(useful_idx)
# print(del2[24])
# print(del2[useful_idx[0],2])
for i in [2,3,-1,-2]:
    idx=0
    for j in range(del2[:,i].size):
        if str(del2[j,i]) =='nan':
            del2[j,i]= del2[useful_idx[idx],i] if re.findall('\d+',str(del2[useful_idx[idx],i])) else 0
        else:
            idx+=1
del3=np.delete(del2,del_idx,0)
useless_idx=[]
for i in range(del3[:,1].size):
    date=del3[i,1].split('-')
    hour=re.findall('(?<=T)[^:]*(?=:)',date[2])[0]
    if date[2].split(':')[1]=='59':
        useless_idx.append(i)
        continue
    else:
        month.append(int(del0(date[1])))
        day.append(del0(date[2].split('T'))[0])
        hours.append(hour)
del4=np.delete(del3,useless_idx,0)
full_data=np.c_[del4,np.array(day)]
full_data=np.c_[full_data,np.array(month)]
full_data=np.c_[full_data,np.array(hours)]
full_data=np.delete(full_data,1,1)
pd.DataFrame(full_data).to_csv("weatherdata.csv")
# np.savetxt('weatherdata.csv',np.asarray(full_data))
# print(full_data)
# final_data=np.delete()
#     # if date[0]== '2019' and date[1]=='01':
    #     if date[2].split(':')[1]=='59':
    #         useful_idx.append(i)
    #         continue
    #     month.append(1)
    #     day.append(del0(date[2].split('T')))
    #     hours.append(hour)
    # elif date[1] in ['06','07','08','09','10','11','12']:
#         if date[2].split(':')[1]=='59':
#             useful_idx.append(i)
#             continue
#         month.append(int(del0(date[1])))
#         day.append(del0(date[2].split('T')))
#         hours.append(hour)
#     else:
#         delete_idx.append(i)
#
# for i in [2,3,-1,-2]:
#     idx=0
#     for j in range(data1[:,i].size):
#         if j not in delete_idx:
#             if str(data1[j,i]) =='nan':
#                 print(data1[useful_idx[idx],i])
#                 data1[j,i]= data1[useful_idx[idx],i] if re.findall('\d+',data1[useful_idx[idx],i]) else 0
#         else:
#             idx+=1
