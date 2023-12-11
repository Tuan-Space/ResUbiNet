import numpy as np

def extract_subseq(seq, site, l):
    s = max(site - l//2 - 1, 0)
    e = min(site + l//2, len(seq))
    return ('-' * (l//2 + 1 - site + s) + seq[s:e] + '-' * (site + l//2 - e)).ljust(l, '-')

def extract_data(dataset_file, protein_file, fasta_33_file, fasta_69_file, label_file):
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
    
    labels = []
    seq_33 = ''
    seq_69 = ''
    
    for line in lines[4:]:
        uniprot_id = line.split('\t')[1]
        
        site = int(line.split('\t')[2][1:])
        label = int(line.split('\t')[3])
        labels.append(label)
        with open(protein_file, 'r') as f:
            plines = f.readlines()
        for i in range(len(plines)):
            if plines[i].startswith(f'>{uniprot_id}'):
                seq = plines[i+1].strip()
                seq_33 += f'>{len(labels):05d}\n'
                seq_33 += extract_subseq(seq, site, 33) + '\n'
                
                seq_69 += f'>{len(labels):05d}\n'
                seq_69 += extract_subseq(seq, site, 69) + '\n'
                break
    
    with open(fasta_33_file, 'w') as f:
        f.write(seq_33)
    
    with open(fasta_69_file, 'w') as f:
        f.write(seq_69)
    
    labels = np.array(labels)
    np.save(label_file, labels)

extract_data('DatasetForhCKSAAP_UbSite/IndependentDataset.txt', 'DatasetForhCKSAAP_UbSite/proteinsForIndependentDataset.txt', 'extracted_datas/independent_33.fasta', 'extracted_datas/independent_69.fasta', 'extracted_datas/independent_labels.npy')
extract_data('DatasetForhCKSAAP_UbSite/TrainingDataset.txt', 'DatasetForhCKSAAP_UbSite/proteinsForTrainingDataset.txt', 'extracted_datas/training_33.fasta', 'extracted_datas/training_69.fasta', 'extracted_datas/training_labels.npy')
