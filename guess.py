import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
import numpy as np
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt  
EPOCH = 100
data = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
my_imputer = SimpleImputer()
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)
train_X = my_imputer.fit_transform(train_X)
train_X, true_X, train_y, true_y = train_test_split(train_X, train_y, test_size=0.1)

test_X = my_imputer.transform(test_X)
var = np.sqrt(np.var(train_y))
mae_list = np.zeros((EPOCH))
p_list_train = np.zeros((EPOCH))
p_list_test = np.zeros((EPOCH))
p_list_clean = np.zeros((EPOCH))

my_model = XGBRegressor()
my_model.fit(train_X, train_y, verbose=False)
predictions = my_model.predict(test_X)
clean = np.corrcoef(predictions,test_y)[0][1]


my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(true_X, true_y, verbose=False)

for i in range(0,EPOCH):
    noise = np.random.normal(loc=0.0, scale= 100000, size=train_y.shape)
    train_Y = train_y + noise  
    
    predictions = my_model.predict(train_X)
    p = np.corrcoef(predictions,train_Y)
    p_list_train[i] = p[0][1]
    final = XGBRegressor()
    sample_weigths = var**2 /(predictions - train_Y)**2
    final.fit(train_X,train_Y,sample_weight = sample_weigths)
    predictions = final.predict(test_X)
    p = np.corrcoef(predictions,test_y)
    p_list_test[i] = p[0][1]


    my_model = XGBRegressor()
    my_model.fit(train_X, train_Y, verbose=False)
    predictions = my_model.predict(test_X)
    p = np.corrcoef(predictions,test_y)
    p_list_train[i] = p[0][1]
    p_list_clean[i] = clean
    # mae = mean_absolute_error(predictions, test_y)
    # mae_list[i] = mae




y = [x for x in range(EPOCH)]
mae_list = mae_list.tolist()
p_list_train = p_list_train.tolist()
p_list_test = p_list_test.tolist()
p_list_clean = p_list_clean.tolist()
l1=plt.plot(y,p_list_train,'r--',label='train')
#l2=plt.plot(y,mae_list,'b--',label='mae')
l3 = plt.plot(y,p_list_test,'b--',label = 'test')
l4 = plt.plot(y,p_list_clean,'y--',label = 'clean')
plt.legend()
plt.show()
