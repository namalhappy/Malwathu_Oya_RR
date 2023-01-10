import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split


data_train = pd.read_csv("..\Dataset\Malwathu\ANFIS\\train_3.csv")
data_test = pd.read_csv("..\Dataset\Malwathu\ANFIS\\test_3.csv")

X, y = data_train.iloc[:,:-1],data_train.iloc[:,-1]
x_train = X
y_train = y


X, y = data_test.iloc[:,:-1],data_test.iloc[:,-1]
x_test = X
y_test = y


xgb_r = xgb.XGBRegressor()
xgb_r.fit(x_train, y_train)

r_score_train = round(xgb_r.score(x_train, y_train),4)
r_score_test = round(xgb_r.score(x_test,y_test),4)

text_r_score_train = 'R = '+ str(r_score_train)
text_r_score_test= 'R = '+str(r_score_test)

print("Train = ",xgb_r.score(x_train, y_train))
print("Test = ",xgb_r.score(x_test,y_test))


## Training accuracy test 
pred = xgb_r.predict(x_train)
pred_t = pred.reshape(-1,1)
y_train_t = y_train.values.reshape(-1,1)

#print(pred_t)

#print(y_train_t)
# p_test = np.append(inv_yhat_tr,inv_yhat)

fig, ax = plt.subplots(facecolor="w")
plt.xlabel("Time Instances",fontsize=15)
plt.ylabel("Water Level (m)",fontsize=15)
#ax.text(0, 5.5, text_r_score_train, fontsize=15,style='italic',
#        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax.plot(pred, label="Prediction")
ax.plot(y_train.values, label="Actual")
ax.legend()
plt.show()

test_values = np.hstack((pred_t,y_train_t))


np.savetxt("Malwathu/train_values_3.csv", test_values, delimiter=",")
# np.savetxt("hand/test_x.csv", X_test, delimiter=",")


## Testing accuracy test 
pred = xgb_r.predict(x_test)
pred_t = pred.reshape(-1,1)
y_train_t = y_test.values.reshape(-1,1)

#print(pred_t)

#print(y_train_t)
# p_test = np.append(inv_yhat_tr,inv_yhat)

fig, ax = plt.subplots(facecolor="w")
plt.xlabel("Time Instances",fontsize=15)
plt.ylabel("Water Level (m)",fontsize=15)
#ax.text(0, 5.5, text_r_score_test, fontsize=15,style='italic',
#        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax.plot(pred, label="Prediction")
ax.plot(y_test.values, label="Actual")
ax.legend()
plt.show()


test_values = np.hstack((pred_t,y_train_t))


np.savetxt("Malwathu/test_values_3.csv", test_values, delimiter=",")
# np.savetxt("hand/test_x.csv", X_test, delimiter=",")

# exit()
fig = plt.figure(figsize=(12,  4))

plt.subplot(1, 2, 1)
plt.xlabel("Time Instances",fontsize=15)
plt.ylabel("Water Level (m)",fontsize=15)
plt.title("Forecast and Actual")
plt.plot(xgb_r.predict(x_test.values), label="Prediction")
plt.plot(y_test.values, label="Actual", alpha=0.8)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Difference")
plt.xlabel("Time Instances",fontsize=15)
plt.plot(y_test.values - xgb_r.predict(x_test.values), label="Actual - Prediction")
plt.legend()

plt.tight_layout()
plt.show()