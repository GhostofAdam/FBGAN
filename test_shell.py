import os

type_list = ["AAC","EAAC","CKSAAP",
            "DPC","DDE","TPC","BINARY","GAAC","EGAAC","CKSAAGP","GDPC","GTPC","AAINDEX","ZSCALE","BLOSUM62",
            "NMBroto","Moran","Geary","CTDC","CTDT","CTDD","CTriad","KSCTriad","SOCNumber","QSOrder","PAAC","APAAC","KNNprotein",
            "KNNpeptide","PSSM","SSEC","SSEB","Disorder","DisorderC","DisorderB","ASA","TA","NAVIE"]

for t in type_list:
    os.system("python3 .\embedding.py "+ t +" 1000 0.001 gbr ")