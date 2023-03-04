import pandas as pd
import numpy as np
train = pd.read_csv("/Users/xinchengzhu/Downloads/train.csv")

#filter missing data and useless columns
train = train[train['MISSING_DATA'] == False] 
train = train[train['POLYLINE'].map(len) > 1] #see line 10
train = train[['POLYLINE']]

#choose first 10000 rows from dataset to run without first row(data type trouble)
train_1 = train.head(10000)
train_1 = train_1[train_1['POLYLINE'] != '[]']

#Change type
for i in range(len(train_1)):
    a = train_1['POLYLINE'].iloc[i]
    a = a[2:-2]
    b = a.split('],[')
    new = []
    for item in b:
        new.append(item.split(','))
    for k in range(len(new)):
        new[k] = [float(j) for j in new[k]]
    a = new

# Normalize MinMax
max_latitude = train_1["POLYLINE"].iloc(0)[0][0][0]
min_latitude = train_1["POLYLINE"].iloc(0)[0][0][0]
max_longtitude = train_1["POLYLINE"].iloc(0)[0][0][1]
min_longtitude = train_1["POLYLINE"].iloc(0)[0][0][1]
for i in range(len(train_1)):
    for cord in train_1["POLYLINE"].iloc(0)[i]:
        max_latitude = max(max_latitude,cord[0])
        max_longtitude = max(max_longtitude,cord[1])
        min_latitude = min(max_latitude,cord[0])
        min_longtitude = min(max_longtitude,cord[1])
print(max_longtitude,max_latitude,min_latitude,min_longtitude)

for i in range(len(train_1)):
    final = []
    for cord in train_1["POLYLINE"].iloc[i]:
        temp = []
        temp.append((cord[0]-min_latitude)/(max_latitude-min_latitude))
        temp.append((cord[1]-min_longtitude)/(max_longtitude-min_longtitude))
        final.append(temp)
    train_1["POLYLINE"].iloc[i] = final

# Split
train_1["POLYLINE_INIT"] = train_1["POLYLINE"]
train_1["POLYLINE_DEST"] = train_1["POLYLINE"]
for i in range(len(train_1)):
    train_1["POLYLINE_INIT"].iloc[i] = train_1["POLYLINE"].iloc[i][:-1]
    train_1["POLYLINE_DEST"].iloc[i] = train_1["POLYLINE"].iloc[i][-1]
train_1_clean = train_1[['POLYLINE_INIT','POLYLINE_DEST']]

#Save to CSV
train_1_clean.to_csv('/Users/xinchengzhu/Downloads/train_1_clean.csv')  
