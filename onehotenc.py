import numpy as np
from sklearn import preprocessing
import pandas as pd
X = pd.read_csv('train_weather.csv')
X_1 = X.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
X_2 = X_1.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
onehotlabels = enc.transform(X_2).toarray()
X=X.drop(['IntersectionId','EntryHeading','ExitHeading','City'], axis=1)
Xtrain=X.values
Xtrain=np.c_[onehotlabels,Xtrain]
np.savetxt('enc_train.csv', Xtrain, delimiter=',')
