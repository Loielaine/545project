from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from numpy.linalg import svd
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics import mean_squared_error
import math
train_data=pd.read_csv("train_sample5.csv")
test_data = pd.read_csv("test_sample5.csv")
train_data=train_data.values
test_data=test_data.values
xtrain= train_data[:,:-3]
ytrain= train_data[:,-3:]
xtest=test_data[:,:-3]
ytest=test_data[:,-3:]
xmean=np.mean(xtrain,0)
xstd=np.std(xtrain,0)
xtrain=(xtrain-xmean)/xstd
xtest=(xtest-xmean)/xstd
pca=PCA(n_components=30)
pc=pca.fit(xtrain).components_
xtrain=(pc.dot(xtrain.T)).T
xtest=(pc.dot(xtest.T)).T
kf=KFold(n_splits=5)
label=[]
mse=[]
# np.linspace(0.01,0.06,6)
for i in [0,1,2]:
    min_error=1e100
    best_r=0
    for c in np.linspace(100,500,5):
        for r in np.linspace(0.01,0.06,6):
            sum_acc=0
            for train_idx, vali_idx in kf.split(xtrain):
                svr_rbf = SVR(kernel='rbf', C=c, gamma=r, epsilon=0.01)
                svr_rbf.fit(xtrain[train_idx,:],ytrain[train_idx,i])
                pre_label=svr_rbf.predict(xtrain[vali_idx,:])
                acc=np.mean((pre_label-ytrain[vali_idx,i])**2)
                sum_acc+=acc
            if sum_acc<min_error:
                min_error=sum_acc
                best_r=r
                bestc=c
            print(sum_acc)
    svr_rbf = SVR(kernel='rbf', C=bestc, gamma=best_r, epsilon=0.01)
    svr_rbf.fit(xtrain,ytrain[:,i])
    pre_label=np.maximum(svr_rbf.predict(xtest),np.zeros(xtest.shape[0]))
    mse.append(mean_squared_error(ytest[:,i], pre_label))
    label.append(pre_label)
fig, axs = plt.subplots(3,1,figsize=(12,10))
axs[0].plot(ytest[:,0],label='p20_test', linewidth=0.8,alpha=0.8)
axs[0].plot(label[0],label='p20_predicted', linewidth=0.8,alpha=0.8,c='r')
axs[1].plot(ytest[:,1],label='p50_test', linewidth=0.8,alpha=0.8)
axs[1].plot(label[1],label='p50_predicted', linewidth=0.8,alpha=0.8,c='r')
axs[2].plot(ytest[:,2],label='p80_test', linewidth=0.8,alpha=0.8)
axs[2].plot(label[2],label='p80_predicted', linewidth=0.8,alpha=0.8,c='r')
axs[0].legend(loc =1,fontsize=12)
axs[1].legend(loc =1,fontsize=12)
axs[2].legend(loc =1,fontsize=12)
axs[0].set_title('Total Time Stopped p20',fontsize=15)
axs[1].set_title('Total Time Stopped p50',fontsize=15)
axs[2].set_title('Total Time Stopped p80',fontsize=15)
axs[0].tick_params(labelsize=12)
axs[1].tick_params(labelsize=12)
axs[2].tick_params(labelsize=12)

plt.text(0.1, 0.9,'RMSE = %f' %np.sqrt(mse[0]), ha='center', va='center', transform=axs[0].transAxes,fontsize=12)
plt.text(0.1, 0.9,'RMSE = %f' %np.sqrt(mse[1]), ha='center', va='center', transform=axs[1].transAxes,fontsize=12)
plt.text(0.1, 0.9,'RMSE = %f' %np.sqrt(mse[2]), ha='center', va='center', transform=axs[2].transAxes,fontsize=12)
# fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
