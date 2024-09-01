import numpy as np
import pandas as pd

aminoacids='ARNDCQEGHILKMFPSTWYV-'
aaindex=pd.read_table('aaindex31',sep='\s+',header=None)
aaindex=aaindex.subtract(aaindex.min(axis=1),axis=0).divide((aaindex.max(axis=1)-aaindex.min(axis=1)),axis=0)
aa=[x for x in 'ARNDCQEGHILKMFPSTWYV']
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
index['-']=np.zeros(31).tolist()
index['X']=np.zeros(31).tolist()
index['Z']=np.zeros(31).tolist()
index['O']=np.zeros(31).tolist()
index['U']=np.zeros(31).tolist()


def index_encode(file):
    encoding=[]
    f=open(file,'r')
    for line in f:
        if not line.startswith('>'):
            col=line.strip()
            encoding.append([index[x] for x in col])
    f.close()
    encoding=np.array(encoding)
    return encoding

for length in range(7, 70, 2):
    aa_train=index_encode(f'../../data_preprocessing/processed_database/train_dataset_{length}.fasta')
    print(aa_train.shape)
    np.save(f'aa_train_{length}.npy', aa_train)

    aa_test=index_encode(f'../../data_preprocessing/processed_database/test_dataset_{length}.fasta')
    print(aa_test.shape)
    np.save(f'aa_test_{length}.npy', aa_test)
