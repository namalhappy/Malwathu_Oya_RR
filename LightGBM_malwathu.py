import pandas as pd
import time


t1 = time.time()

# data = pd.read_csv("..\Dataset\Malwathu\malwathu_1_day_NH.csv",header=None)
# X, y = data.iloc[:,:-1],data.iloc[:,-1]
#print(data.head(3))

data_train = pd.read_csv("..\Dataset\Malwathu\ANFIS\\train_3.csv")
data_test = pd.read_csv("..\Dataset\Malwathu\ANFIS\\test_3.csv")

X, y = data_train.iloc[:,:-1],data_train.iloc[:,-1]
X_train = X
y_train = y


X, y = data_test.iloc[:,:-1],data_test.iloc[:,-1]
X_test = X
y_test = y

import os
import random
import numpy as np
from matplotlib import pyplot as plt


import lightgbm as lgb
from sklearn.model_selection import train_test_split

SEED = 42

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)


train_values.to_csv("Malwathu/train_1.csv",index=False,header = False)  

train_values = X_test

train_values['out'] = y_test

train_values.to_csv("Malwathu/test_1.csv",index=False,header = False)  

lgb_train = lgb.Dataset(X_train.values, y_train.values)
lgb_test = lgb.Dataset(X_test.values, y_test.values, reference=lgb_train)                            
                            
params = {
          'task': 'train', 
          'boosting_type': 'gbdt',    
          'objective': 'regression',    
          'metric': 'rmse',  
          'learning_rate': 0.01, 
          }                            
                            
lgb_results = {}                                    


model = lgb.train(
                  params=params,                    
                  train_set=lgb_train,              
                  valid_sets=[lgb_train, lgb_test], 
                  valid_names=['Train', 'Test'],   
                  num_boost_round=100,              
                  early_stopping_rounds=50,        
                  evals_result=lgb_results,           
                  verbose_eval=-1                          
                  )                             

pred_t = model.predict(X_train.values)
pred_t = pred_t.reshape(-1,1)
y_train_t = y_train.values.reshape(-1,1)
train_values = np.hstack((pred_t,y_train_t))


np.savetxt("Malwathu/train_values_3.csv", train_values, delimiter=",")
# np.savetxt("hand/test_x.csv", X_test, delimiter=",")

fig, ax = plt.subplots(facecolor="w")
plt.xlabel("Time Instances",fontsize=15)
plt.ylabel("Water Level (m)",fontsize=15)
#ax.text(0, 5.5, text_r_score_train, fontsize=15,style='italic',
        #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax.plot(pred_t, label="Prediction")
ax.plot(y_train.values, label="Actual")
ax.legend()
plt.show()

pred_t = model.predict(X_test.values)
pred_t = pred_t.reshape(-1,1)
y_test_t = y_test.values.reshape(-1,1)
test_values = np.hstack((pred_t,y_test_t))


np.savetxt("Malwathu/test_values_3.csv", test_values, delimiter=",")
# np.savetxt("hand/test_x.csv", X_test, delimiter=",")

fig, ax = plt.subplots(facecolor="w")
plt.xlabel("Time Instances",fontsize=15)
plt.ylabel("Water Level (m)",fontsize=15)
#ax.text(0, 5.5, text_r_score_train, fontsize=15,style='italic',
        #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax.plot(pred_t, label="Prediction")
ax.plot(y_test.values, label="Actual")
ax.legend()
plt.show()


