import numpy as np
import re
import pandas as pd
data1 = pd.read_csv("weatherdata.csv")
data1=data1.values
print(type(data1[6,3]))
for i in range(data1.shape[0]):
    for j in range(data1.shape[1]):
        if str(data1[i,j])=='nan':
            pass
        else:
            if not re.findall('\d+',str(data1[i,j])):
                data1[i,j]=0
            else:
                if '.' in str(data1[i,j]):
                    data1[i,j]=float(re.findall('\d+\.\d+',str(data1[i,j]))[0])
                else:
                    data1[i,j]=float(re.findall('\d+',str(data1[i,j]))[0])
pd.DataFrame(data1).to_csv("weather_data.csv")
