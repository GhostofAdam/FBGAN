import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
import numpy as np
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt  

data = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)

my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
var = np.sqrt(np.var(train_y))
# mae_list = np.zeros((100))
# p_list = np.zeros((100))
# for i in range(0,100):
#     noise = np.random.normal(loc=0.0, scale= i*1000, size=train_y.shape)
#     train_Y = train_y + noise  
#     my_model = XGBRegressor()
#     # Add silent=True to avoid printing out updates with each cycle
#     my_model.fit(train_X, train_Y, verbose=False)

#     predictions = my_model.predict(test_X)
#     p = np.corrcoef(predictions,test_y)
#     p_list[i] = p[0][1]
#     mae = mean_absolute_error(predictions, test_y)
#     mae_list[i] = mae
#     #print("Mean Absolute Error : " + str(mae))
# print(p_list)
# y = [x for x in range(0,100000,1000)] / var
# np.save("mae_list.npy",mae_list)
# np.save("p_list.npy",p_list)
# mae_list = mae_list.tolist()
# p_list = p_list.tolist()

# #l1=plt.plot(y,p_list,'r--',label='p')
# l2=plt.plot(y,mae_list,'b--',label='mae')

# plt.legend()
# plt.show()
mae_list = np.zeros((100))
p_list = np.zeros((100))
for i in range(0,100):
    noise = np.random.normal(loc=0.0, scale= 100000, size=train_y.shape)
    train_y = train_y + noise  
    my_model = XGBRegressor()
    # Add silent=True to avoid printing out updates with each cycle
    my_model.fit(train_X, train_y, verbose=False)

    predictions = my_model.predict(test_X)
    p = np.corrcoef(predictions,test_y)
    p_list[i] = p[0][1]
    mae = mean_absolute_error(predictions, test_y)
    mae_list[i] = mae
    #print("Mean Absolute Error : " + str(mae))
print(p_list)
y = [x for x in range(100)]
np.save("mae_list.npy",mae_list)
np.save("p_list.npy",p_list)
mae_list = mae_list.tolist()
p_list = p_list.tolist()

l1=plt.plot(y,p_list,'r--',label='p')
#l2=plt.plot(y,mae_list,'b--',label='mae')

plt.legend()
plt.show()
# my_model = XGBRegressor(n_estimators=1000)
# my_model.fit(train_X, train_y, early_stopping_rounds=5, 
#              eval_set=[(test_X, test_y)], verbose=False)

# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(train_X, train_y, early_stopping_rounds=5, 
#              eval_set=[(test_X, test_y)], verbose=False)