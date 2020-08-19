
import pandas as pd
import numpy as np
from loader import *
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn import tree
from sklearn import svm
import argparse
import matplotlib.pyplot as plt  

def main(embed_type = "AAC",EPOCH=2000,LR =1e-14,mode = "knn",output = None):

    train_X, train_Y, test_X, test_Y = data_loader(embed_type)

    if mode == "adaboost":
        regr = AdaBoostRegressor(learning_rate=LR,loss='linear',random_state=0, n_estimators=EPOCH)
    elif mode == "knn":
         regr = KNeighborsRegressor(weights='uniform')
    elif mode == "tree":
        regr = tree.DecisionTreeRegressor()
    elif mode == "svr":
        regr = svm.SVR()
    elif mode == "gbr":
        regr = GradientBoostingRegressor(n_estimators=EPOCH,learning_rate=LR,loss="huber")
   
    print("start training")
    regr.fit(train_X, train_Y)
    pred = regr.predict(test_X)
    print(pred.shape)
    print(np.corrcoef(pred,test_Y))
    p = np.corrcoef(pred,test_Y)
    pred = pred.tolist()
    test_Y = test_Y.tolist()
    l1=plt.scatter(test_Y,pred)
    plt.show()
    if output !=None:
        np.save(output+".txt",pred)
    with open("res.txt","a") as f:
        f.write(embed_type+" "+str(p[0][1])+"\n")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("type",type = str, help="embedding type")
    parser.add_argument("EPOCH",type = int, help="epoch")
    parser.add_argument("LR",type=float, help="learning rate")
    parser.add_argument("mode",type=str, help="algorithm")
    #parser.add_argument("output",type=str, help="output path")
    args=parser.parse_args()
    main(args.type,args.EPOCH,args.LR,args.mode,args.type)
        
        
