import pandas as pd
import numpy as np
def data_loader(embed_type):
    data = pd.read_csv("data.csv",index_col=0)
    if embed_type == "NAVIE":
        X = np.zeros((len(data),26))
        for i, seq in enumerate(data['i']):
            for letter in seq:
                X[i][ord(letter)-ord('A')] += 1
    else:
        embedding =  pd.read_csv("./iFeature/iFeature/"+embed_type+".tsv",sep='\t')
        X = embedding.iloc[:,1:].to_numpy()
    
    Y = data['c'].to_numpy()
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

    train_X = X
    train_Y = Y
    df = pd.read_excel("thermo_tyrosineRS.xlsx",index_col=0)
    test_Y = df[['Optimal growth temperature']].to_numpy().squeeze()
    if embed_type == "NAVIE":
        test_X = np.zeros((len(df),26))
        for i, seq in enumerate(df['organism']):
            for letter in seq:
                X[i][ord(letter)-ord('A')] += 1
    else:
        embedding =  pd.read_csv("./iFeature/iFeature/"+embed_type+"_test.tsv",sep='\t')
        test_X = embedding.iloc[:,1:].to_numpy()
    
    return train_X,train_Y,test_X,test_Y

if __name__ == "__main__":
    Y,X = data_loader()
    print(X[0])