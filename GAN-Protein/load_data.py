import xlrd
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
BATCH_SIZE = 50

# dictionary
charmap = {'G': 0, 'S': 1, 'A': 2, 'T': 3, 'V': 4, 'I': 5, 'L': 6, 'Y': 7, 'F': 8, 'H': 9,
           'P': 10, 'D': 11, 'M': 12, 'E': 13, 'W': 14, 'K': 15, 'C': 16, 'R': 17, 'N': 18, 'Q': 19, 'B': 20}
inv_charmap = ['G', 'S', 'A', 'T', 'V', 'I', 'L', 'Y', 'F', 'H',
               'P', 'D', 'M', 'E', 'W', 'K', 'C', 'R', 'N', 'Q', 'B']


# Dataset iterator
def data_real_gen(lines):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield lines[i:i+BATCH_SIZE]

    # np.array([[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]], dtype='int32')


# Load data
def LoadData():
    Data = xlrd.open_workbook('tyrosine_thermostability/uniprot-tyrosine+trna+synthase+bacterium.xlsx')
    sheet = Data.sheet_by_name('Sheet0')
    old_X = sheet.col_values(-1, 0)
    Len = sheet.col_values(-2, 0)
    X = []
    for i in range(1, len(old_X)):
        if Len[i] <= 450 and old_X[i].find('X') == -1 and old_X[i].find('U') == -1:
            X.append(old_X[i])
    for i in range(len(X)):
        X[i] += (450-len(X[i]))*'B'
    # res = pd.DataFrame({'X_filter': X})
    # res.to_excel('X_filter.xlsx')
    X_output = np.zeros([len(X), 450, 21], dtype=int)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X_output[i][j][charmap[X[i][j]]] = 1
    # X_tsne = X_output.reshape([-1, 9450])
    # for i in range(len(X_tsne)):
    #     counts = 0
    #     for j in range(len(X_tsne[i])):
    #         if X_tsne[i][j] == 1:
    #             counts += 1
    #     print(counts)
    return X_output


# test
if __name__ == '__main__':
    data = LoadData()
    # tsne = TSNE(n_components=2)
    # tsne.fit_transform(data)
    # print(tsne.embedding_)
