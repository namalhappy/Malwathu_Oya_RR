import pandas as pd
import time


t1 = time.time()



data_train = pd.read_csv("..\Dataset\Malwathu\ANFIS\\WH\\train_3.csv")
data_test = pd.read_csv("..\Dataset\Malwathu\ANFIS\\WH\\test_3.csv")

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


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

SEED = 42

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

train_values = X_train

train_values['out'] = y_train
train_values.to_csv("Malwathu/train_3.csv",index=False,header = False)  

train_values = X_test

train_values['out'] = y_test
train_values.to_csv("Malwathu/test_3.csv",index=False,header = False)  

CAT_FEATURES = []

model = CatBoostRegressor(n_estimators=100,
                                   loss_function = 'RMSE',
                                   eval_metric = 'RMSE',
                                   cat_features = CAT_FEATURES)

# fit model
model.fit(X_train, y_train, 
                   eval_set = (X_test, y_test),
                   use_best_model = True,
                   plot = True)

# exit()


## Seperate prediciton plots for train and test

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


