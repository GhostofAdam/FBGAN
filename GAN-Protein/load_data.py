import xlrd
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
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
    # Data = xlrd.open_workbook('tyrosine_thermostability/uniprot-tyrosine+trna+synthase+bacterium.xlsx')
    # Data = xlrd.open_workbook('tyrosine_thermostability/uniprot-tyrosine+trna+synthase+fungi.xlsx')
    # sheet = Data.sheet_by_name('Sheet0')
    # old_X = sheet.col_values(-1, 0)
    # Len = sheet.col_values(-2, 0)
    # X = []
    # for i in range(1, len(old_X)):
    #     if Len[i] <= 450 and old_X[i].find('X') == -1 and old_X[i].find('U') == -1:
    #         X.append(old_X[i])
    # for i in range(len(X)):
    #     X[i] += (450-len(X[i]))*'B'
    # res = pd.DataFrame({'X_filter': X})
    # res.to_excel('X_filter.xlsx')
    with open('samples_10019.txt', 'r') as myfile:
        X = []
        for line in myfile:
            zj = ''
            for c in line:
                if c != '\n' and c != 'B':
                    zj += c
            X.append(zj)
    X_output = np.zeros([len(X), 450, 21], dtype=int)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X_output[i][j][charmap[X[i][j]]] = 1
    X_tsne = X_output.reshape([-1, 450*21])
    return X_tsne, X_output


# Plot t-SNE
def plot_embedding(data, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # model = KMeans(n_clusters=3).fit(data)
    # labels = DBSCAN(eps=0.02, min_samples=1).fit_predict(data)
    # print(labels)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print(n_clusters_)
    # col = ['bo', 'go', 'ro', 'co', 'mo', 'yo',
    #        'b*', 'g*', 'r*', 'c*', 'm*', 'y*',
    #        'bv', 'gv', 'rv', 'cv', 'mv', 'yv',
    #        'bd', 'gd', 'rd', 'cd', 'md', 'yd',
    #        'b^', 'g^', 'r^', 'c^', 'm^', 'y^',
    #        'b+', 'g+', 'r+', 'c+', 'm+', 'y+'
    #        ]
    # labels = model.labels_
    plt.figure()
    # for i in range(data.shape[0]):
    #     if labels[i] == -1:
    #         plt.plot(data[i, 0], data[i, 1], 'ko')
    #     elif labels[i] > 35:
    #         plt.plot(data[i, 0], data[i, 1], 'k*')
    #     else:
    #         plt.plot(data[i, 0], data[i, 1], col[labels[i]])
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], 'bo')
    plt.title(title)
    plt.show()



def transform_fasta():
    Data = xlrd.open_workbook('tyrosine_thermostability/uniprot-tyrosine+trna+synthase+bacterium.xlsx')
    sheet = Data.sheet_by_name('Sheet0')
    old_X = sheet.col_values(-1, 0)
    Len = sheet.col_values(-2, 0)
    data_id = sheet.col_values(0, 0)
    X = []
    X_id = []
    for i in range(1, len(old_X)):
        if Len[i] <= 450 and old_X[i].find('X') == -1 and old_X[i].find('U') == -1:
            X.append(old_X[i])
            X_id.append(data_id[i])

    with open('myseq.txt', 'w') as myfile:
        for i in range(len(X)):
            myfile.write('>'+X_id[i]+'\n'+X[i]+'\n')


# test
if __name__ == '__main__':
    X_tsne, X_output = LoadData()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2)
    t0 = time()
    result = tsne.fit_transform(X_tsne[:])
    plot_embedding(result, 't-SNE embedding of the data (time %.2fs)' % (time() - t0))
    # transform_fasta()
