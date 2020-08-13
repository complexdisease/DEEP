import math
import sys
import pyfasta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import math
eps=1E-6
dir= $DIR
bedfile=dir+$VCF
bed=pd.read_csv(bedfile, sep='\t', header=None, comment='#')
bed.iloc[:, 0] = 'chr' + bed.iloc[:, 0].map(str).str.replace('chr', '')
Batchsize=500
genome = pyfasta.Fasta(dir+'./Index/hg19.fa')
windowsize=2000.0
binsize=20.0
nfeatures=1

CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9','chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4,
                      out_channels=320, kernel_size=8), 

            nn.ReLU(),
            nn.Conv1d(320, 320, 8),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(320, 480, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(480, 480, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(4),
            nn.Conv1d(480, 640, 8), nn.ReLU(), nn.Conv1d(640, 640, 8),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(106 * 640, 2000), nn.ReLU(), nn.Linear(2000, 120), nn.ReLU(), nn.Linear(120, 4),nn.Sigmoid()
        )
    def forward(self,t):
        t=self.conv1(t)
        t=self.conv2(t)
        t=t.view(t.size(0),-1)
        t=self.fc(t)
        return t

def Seq_Coding(seqs, inputsize):
    seqsnp = np.zeros((len(seqs), 4, inputsize))
    dict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),'C': np.asarray([0, 0, 1, 0]),
            'T': np.asarray([0, 0, 0, 1]),'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),'a': np.asarray([1, 0, 0, 0]),
            'g': np.asarray([0, 1, 0, 0]), 'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]), 'n': np.asarray([0, 0, 0, 0]),
            '-': np.asarray([0, 0, 0, 0])}
    n=0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = dict[c]
        n = n + 1
    flip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, data], axis=0)
    return seqsnp

os.mkdir(dir+"out/")
cnn=CNN()
cnn.load_state_dict(torch.load(dir+'model/atac_PG.pth'))
cnn.eval()
if torch.cuda.is_available():
    torch.cuda.get_device_name(0)
    cnn.cuda()
for shift in np.arange(0,windowsize+1,binsize):
    print(shift)
    ref_seqlist, mut_seqlist ,match_ref_list = [], [], []
    distance=shift-windowsize/2
    for i in range(bed.shape[0]):
        seq = genome.sequence({'chr': bed.iloc[i, 0], 'start': bed.iloc[i, 1]-windowsize+shift, 'stop': bed.iloc[i, 1]+shift})
        position = windowsize - shift
        ref,alt=usedbed.iloc[i,2],usedbed.iloc[i,3]
        altseq = seq[:position]+alt+seq[position+len(ref):]
        refseq = seq[:position]+ref+seq[position+len(ref):]
        match_ref=seq[position:(position + len(ref))].upper() == ref.upper()
        ref_seqlist.append(refseq)
        alt_seqlist.append(altseq)
        match_ref_list.append(match_ref)
    if shift == 0:
        print('Matched REF:\t', np.sum(match_ref_list), 'Total Sites:\t', len(match_ref_list))
    ref_encoded = Seq_Coding(ref_seqlist, inputsize=windowsize).astype(np.float32)
    alt_encoded = Seq_Coding(alt_seqlist, inputsize=windowsize).astype(np.float32)
    ref_preds = []
    for i in range(int(1 + (ref_encoded.shape[0] - 1) / Batchsize)):
        input = torch.from_numpy(ref_encoded[int(i * Batchsize):int((i + 1) * Batchsize), :, :]).cuda()
        ref_preds.append(cnn.forward(input).cpu().detach().view(-1,8).numpy().copy())
    ref_preds = np.vstack(ref_preds)
    alt_preds = []
    for i in range(int(1 + (alt_encoded.shape[0] - 1) / Batchsize)):
        input = torch.from_numpy(alt_encoded[int(i * Batchsize):int((i + 1) * Batchsize), :, :]).cuda()
        alt_preds.append(cnn.forward(input).cpu().detach().view(-1,8).numpy().copy())
    alt_preds = np.vstack(alt_preds)
    diff = alt_preds - ref_preds
    output=np.concatenate((ref_preds,alt_preds),axis=1)
    match_ref_list=np.array(match_ref_list).reshape(-1,1)
    match_ref_list=np.vstack((match_ref_list,match_ref_list))
    output=np.concatenate((match_ref_list,output),axis=1)
    pos,neg=bed,bed
    pos['strand']='pos'
    neg['strand']='neg'
    stack=pd.concat((pos,neg),axis=0,ignore_index=True).to_numpy()
    stack=np.concatenate((stack,output),axis=1)
    np.save(dir+str(shift)+'.npy',stack)

filelist=os.listdir(dir+'out/')
bedlist=[]
weighted_diff0,weighted_diff1=pd.DataFrame(),pd.DataFrame()
ref_NP,refB,altA,altB=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
header=['CHR','POS','REF','ALT','Strand','REF_MATCHED']
weights=[]
feature_list,output=list(),list()
for k in range(nfeatures*2):
    feature_list.append(pd.DataFrame())
for i,file in enumerate(filelist):
    if file.endswith("npy"):
        loc=windowsize-float(file.split('.')[0])
        dist=max(abs(loc-windowsize/2)-binsize/2,0.0)
        print(loc,dist)
        weight=10*math.exp(-1*dist/(1*binsize))
        weights.append(weight)
        bed=np.load(dir+'out/'+file,allow_pickle=True)
        if i==0:
            ann=bed.iloc[:,:5]
            ann.columns=header
        bedlist.append(bed.iloc[:,5:-1])
        for k in range(nfeatures):
            diff_ref=np.log2(bed.iloc[:,2*k+1])-np.log2(1-bed.iloc[:,2*k+1])-np.log2(bed.iloc[:,2*k])+np.log2(1-bed.iloc[:,2*k])
            diff_alt=np.log2(bed.iloc[:,2*(k+nfeatures)+1])-np.log2(1-bed.iloc[:,2*(k+nfeatures)+1])-np.log2(bed.iloc[:,2*(k+nfeatures)])+np.log2(1-bed.iloc[:,2*(k+nfeatures)])
            feature_list[k][str(loc)]=(diff_alt-diff_ref)*weight
score=pd.DataFrame()
for k in range(nfeatures):
    score[k]=feature_list[k].sum(axis=1)
score=pd.concat((ann,score),axis=1,ignore_index=True)
score.to_csv(dir+'DEEP_output.txt',sep='\t',header=True,index=False)





