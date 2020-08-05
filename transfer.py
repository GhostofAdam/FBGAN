import pandas as pd
import re
import fastaparser

df = pd.read_excel("thermo_tyrosineRS.xlsx",index_col=0)
df = df[['Optimal growth temperature','organism']]
print(df)
seq = open("seq_test.fasta","w")
writer = fastaparser.Writer(seq)
label = open("label_test.txt","w+")
i = 0
before = 0
for index, row in (df.iterrows()):
    
    writer.writefasta((str(index)+str(i),row['organism']))

    #label.write(str(index)+str(i)+" "+str(row['organism'])+"\n")
    i+=1
    before = row['Optimal growth temperature']
seq.close()
label.close()


