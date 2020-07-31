import numpy as np
import matplotlib.pyplot as plt

# Construct dictionary
bi_index = {'G': 0, 'S': 1, 'A': 2, 'T': 3, 'V': 4,
            'I': 5, 'L': 6, 'Y': 7, 'F': 8, 'H': 9,
            'P': 10, 'D': 11, 'M': 12, 'E': 13, 'W': 14,
            'K': 15, 'C': 16, 'R': 17, 'N': 18, 'Q': 19}

# open file
# Generated samples
with open('samples_10019.txt', 'r') as myfile:
    samples = []
    for line in myfile:
        zj = ''
        for c in line:
            if c != 'B' and c != '\n':
                zj += c
        samples.append(zj)
# Natural samples
# with open('myseq.txt', 'r') as myfile:
#     samples = []
#     for line in myfile:
#         zj = ''
#         if line[0] != '>':
#             for c in line:
#                 if c != 'B' and c != '\n':
#                     zj += c
#             samples.append(zj)
# print(samples[0])

# Extract data
bi_counts = np.zeros([400], dtype=float)
for i in range(len(samples)):
    for j in range(0, len(samples[i])-1):
        bi_counts[bi_index[samples[i][j]] * 20 + bi_index[samples[i][j + 1]]] += 1
counts_mean = np.mean(bi_counts)
counts_std = np.std(bi_counts, ddof=1)
counts = bi_counts.reshape([20, 20])
counts_new = np.zeros([20, 20], dtype=float)
for i in range(len(counts)):
    for j in range(len(counts[i])):
        counts_new[19-i][j] = (counts[i][j] - counts_mean) / counts_std

# print(counts)
# print(counts_new)


# Setting the x/y axis
xLabel = ['G', 'S', 'A', 'T', 'V', 'I', 'L', 'Y', 'F', 'H',
          'P', 'D', 'M', 'E', 'W', 'K', 'C', 'R', 'N', 'Q']
yLabel = ['G', 'S', 'A', 'T', 'V', 'I', 'L', 'Y', 'F', 'H',
          'P', 'D', 'M', 'E', 'W', 'K', 'C', 'R', 'N', 'Q']
yLabel.reverse()

# Plot heatmap
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yticks(range(len(yLabel)))
ax.set_yticklabels(yLabel)
ax.set_xticks(range(len(xLabel)))
ax.set_xticklabels(xLabel)
plt.xlabel('Amino acid(Second)')
plt.ylabel('Amino acid(First)')
im = ax.imshow(counts_new, cmap=plt.cm.hot_r)
plt.colorbar(im)
plt.title("Zm(Generated) 10019")
plt.show()





