#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

import math
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import sklearn
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import argparse

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format

parser = argparse.ArgumentParser()
parser.add_argument("-srcPath", help="Source Path for original data: Raw Format", type=str, default="/scratch/work/project/excape-public/data_release_v5/version5/")
parser.add_argument("-destPath", help="Dest Path for original data:", type=str, default=os.getenv("HOME")+"/mydatasets/excapedb/current2/")
args = parser.parse_args()




dataPath=args.srcPath
dataPathSave=args.destPath
if not os.path.exists(dataPathSave):
  os.makedirs(dataPathSave)



folding=pd.read_csv(dataPath+"folds.txt.gz", sep="\t", header=0, index_col=0)
clustering=pd.read_csv(dataPath+"clustering.txt.gz", sep="\t", header=0, index_col=None).drop_duplicates()
sortedClustering=clustering.sort_values(by='cluster')
compoundNames=sortedClustering.iloc[:,0].values



for foldIndex in range(0,3):
  fold1Clusters=folding.loc[folding.iloc[:,foldIndex]=='c1'].index.values
  fold2Clusters=folding.loc[folding.iloc[:,foldIndex]=='c2'].index.values
  fold3Clusters=folding.loc[folding.iloc[:,foldIndex]=='c3'].index.values
  
  indFold1Left=sortedClustering.iloc[:,1].values.searchsorted(fold1Clusters,side='left')
  indFold1Right=sortedClustering.iloc[:,1].values.searchsorted(fold1Clusters,side='right')
  fold1Samples=[y for x in zip(indFold1Left, indFold1Right) for y in compoundNames[x[0]:x[1]].tolist()]
  
  indFold2Left=sortedClustering.iloc[:,1].values.searchsorted(fold2Clusters,side='left')
  indFold2Right=sortedClustering.iloc[:,1].values.searchsorted(fold2Clusters,side='right')
  fold2Samples=[y for x in zip(indFold2Left, indFold2Right) for y in compoundNames[x[0]:x[1]].tolist()]
  
  indFold3Left=sortedClustering.iloc[:,1].values.searchsorted(fold3Clusters,side='left')
  indFold3Right=sortedClustering.iloc[:,1].values.searchsorted(fold3Clusters,side='right')
  fold3Samples=[y for x in zip(indFold3Left, indFold3Right) for y in compoundNames[x[0]:x[1]].tolist()]
  
  folds=[fold1Samples, fold2Samples, fold3Samples]
  f=open(dataPathSave+'folds'+str(foldIndex)+'.pckl', "wb")
  pickle.dump(folds, f, -1)
  f.close()



annotations=pd.read_csv(dataPath+"activities.txt.gz", header=0, sep="\t").drop_duplicates()
sampleAnnInd=pd.Series(data=np.arange(len(np.unique(annotations.iloc[:,0].values))), index=np.unique(annotations.iloc[:,0].values))
targetAnnInd=pd.Series(data=np.arange(len(np.unique(annotations.iloc[:,1].values))), index=np.unique(annotations.iloc[:,1].values))
annMat=scipy.sparse.coo_matrix((annotations.iloc[:,2], (sampleAnnInd[annotations.iloc[:,0]], targetAnnInd[annotations.iloc[:,1]])), shape=(sampleAnnInd.max()+1, targetAnnInd.max()+1))
annMat=annMat.tocsr()
annMat.sort_indices()
f=open(dataPathSave+'values.pckl', "wb")
pickle.dump(annMat, f, -1)
pickle.dump(sampleAnnInd, f, -1)
pickle.dump(targetAnnInd, f, -1)
f.close()

annMat1=annMat.copy().astype(np.int64)
annMat1.data[:]=0
annMat1.data[(annMat.data<5.0)]=-1
annMat1.data[(annMat.data>=5.0)]=1
annMat1.eliminate_zeros()
annMat1.tocsr()
annMat1.sort_indices()
annMat2=annMat.copy().astype(np.int64)
annMat2.data[:]=0
annMat2.data[(annMat.data<6.0)]=-1
annMat2.data[(annMat.data>=6.0)]=1
annMat2.eliminate_zeros()
annMat2.tocsr()
annMat2.sort_indices()
annMat3=annMat.copy().astype(np.int64)
annMat3.data[:]=0
annMat3.data[(annMat.data<7.0)]=-1
annMat3.data[(annMat.data>=7.0)]=1
annMat3.eliminate_zeros()
annMat3.tocsr()
annMat3.sort_indices()
annMat4=annMat.copy().astype(np.int64)
annMat4.data[:]=0
annMat4.data[(annMat.data<8.0)]=-1
annMat4.data[(annMat.data>=8.0)]=1
annMat4.eliminate_zeros()
annMat4.tocsr()
annMat4.sort_indices()
annMatAll=scipy.sparse.hstack([annMat1, annMat2, annMat3, annMat4])
annMatAll.eliminate_zeros()
annMatAll=annMatAll.tocsr()
annMatAll.sort_indices()
sampleAnnIndAll=sampleAnnInd.copy()
targetAnnIndAll=pd.Series(data=np.arange(len(targetAnnInd)*4), index=np.concatenate(['t1_trg_'+targetAnnInd.index.values, 't2_trg_'+targetAnnInd.index.values, 't3_trg_'+targetAnnInd.index.values, 't4_trg_'+targetAnnInd.index.values]))
f=open(dataPathSave+'labelsHard.pckl', "wb")
pickle.dump(annMatAll, f, -1)
pickle.dump(sampleAnnIndAll, f, -1)
pickle.dump(targetAnnIndAll, f, -1)
f.close()



annotationLevels=pd.read_csv(dataPath+"activities_levels.txt.gz", header=0, sep="\t").drop_duplicates()
sampleAnnLevelInd=pd.Series(data=np.arange(len(np.unique(annotationLevels.iloc[:,0].values))), index=np.unique(annotationLevels.iloc[:,0].values))
targetAnnLevelInd=pd.Series(data=np.arange(len(np.unique(annotationLevels.iloc[:,1].values))), index=np.unique(annotationLevels.iloc[:,1].values))
annMatLevel=scipy.sparse.coo_matrix((annotationLevels.loc[:,"isge6"].values, (sampleAnnLevelInd[annotationLevels.iloc[:,0]], targetAnnLevelInd[annotationLevels.iloc[:,1]])), shape=(sampleAnnLevelInd.max()+1, targetAnnLevelInd.max()+1))
annMatLevel=annMatLevel.tocsr()
annMatLevel.sort_indices()
f=open(dataPathSave+'labelsExCAPE.pckl', "wb")
pickle.dump(annMatLevel, f, -1)
pickle.dump(sampleAnnLevelInd, f, -1)
pickle.dump(targetAnnLevelInd, f, -1)
f.close()



chem2vec=pd.read_csv(dataPath+"chem2vec.txt.gz", sep="\t", header=None, index_col=0)
assert(len(chem2vec.index.values)==len(np.unique(chem2vec.index.values)))
assert(len(chem2vec.columns.values)==len(np.unique(chem2vec.columns.values)))
chem2vecMat=chem2vec.values.copy()
sampleChem2VecInd=pd.Series(data=np.arange(len(chem2vec.index.values)), index=chem2vec.index.values)
featureChem2VecInd=pd.Series(data=np.arange(len(chem2vec.columns.values)), index=chem2vec.columns.values)
f=open(dataPathSave+'chem2vec.pckl', "wb")
pickle.dump(chem2vecMat, f, -1)
pickle.dump(sampleChem2VecInd, f, -1)
pickle.dump(featureChem2VecInd, f, -1)
f.close()



ecfp6=pd.read_csv(dataPath+"ecfp6_counts.txt.gz", sep="\t", header=0, index_col=None).drop_duplicates()
sampleECFP6Ind=pd.Series(data=np.arange(len(np.unique(ecfp6.iloc[:,0].values))), index=np.unique(ecfp6.iloc[:,0].values))
featureECFP6Ind=pd.Series(data=np.arange(len(np.unique(ecfp6.iloc[:,1].values))), index=np.unique(ecfp6.iloc[:,1].values))
ecfp6Mat=scipy.sparse.coo_matrix((ecfp6.iloc[:,2], (sampleECFP6Ind[ecfp6.iloc[:,0]], featureECFP6Ind[ecfp6.iloc[:,1]])), shape=(sampleECFP6Ind.max()+1, featureECFP6Ind.max()+1))
ecfp6Mat=ecfp6Mat.tocsr()
ecfp6Mat.sort_indices()
f=open(dataPathSave+'ecfp6.pckl', "wb")
pickle.dump(ecfp6Mat, f, -1)
pickle.dump(sampleECFP6Ind, f, -1)
pickle.dump(featureECFP6Ind, f, -1)
f.close()



ecfp6Folded=pd.read_csv(dataPath+"ecfp6_folded.txt.gz", sep="\t", header=0, index_col=0)
assert(len(ecfp6Folded.index.values)==len(np.unique(ecfp6Folded.index.values)))
assert(len(ecfp6Folded.columns.values)==len(np.unique(ecfp6Folded.columns.values)))
ecfp6FoldedMat=ecfp6Folded.values.copy()
sampleECFP6FoldedInd=pd.Series(data=np.arange(len(ecfp6Folded.index.values)), index=ecfp6Folded.index.values)
featureECFP6FoldedInd=pd.Series(data=np.arange(len(ecfp6Folded.columns.values)), index=ecfp6Folded.columns.values)
f=open(dataPathSave+'ecfp6Folded.pckl', "wb")
pickle.dump(ecfp6FoldedMat, f, -1)
pickle.dump(sampleECFP6FoldedInd, f, -1)
pickle.dump(featureECFP6FoldedInd, f, -1)
f.close()



ecfp6Var005=pd.read_csv(dataPath+"ecfp6_counts_var005.txt.gz", sep="\t", header=0, index_col=None).drop_duplicates()
sampleECFP6Var005Ind=pd.Series(data=np.arange(len(np.unique(ecfp6Var005.iloc[:,0].values))), index=np.unique(ecfp6Var005.iloc[:,0].values))
featureECFP6Var005Ind=pd.Series(data=np.arange(len(np.unique(ecfp6Var005.iloc[:,1].values))), index=np.unique(ecfp6Var005.iloc[:,1].values))
ecfp6Var005Mat=scipy.sparse.coo_matrix((ecfp6Var005.iloc[:,2], (sampleECFP6Var005Ind[ecfp6Var005.iloc[:,0]], featureECFP6Var005Ind[ecfp6Var005.iloc[:,1]])), shape=(sampleECFP6Var005Ind.max()+1, featureECFP6Var005Ind.max()+1))
ecfp6Var005Mat=ecfp6Var005Mat.tocsr()
ecfp6Var005Mat.sort_indices()
f=open(dataPathSave+'ecfp6Var005.pckl', "wb")
pickle.dump(ecfp6Var005Mat, f, -1)
pickle.dump(sampleECFP6Var005Ind, f, -1)
pickle.dump(featureECFP6Var005Ind, f, -1)
f.close()



#ecfp6Mat2=ecfp6Mat.copy()
#ecfp6Mat2.data[:]=ecfp6Mat2.data*ecfp6Mat2.data
#featSelVar005=((ecfp6Mat2.sum(0).A[0].astype(np.float64)/ecfp6Mat2.getnnz(0).astype(np.float64))-(ecfp6Mat.sum(0).A[0].astype(np.float64)/ecfp6Mat2.getnnz(0).astype(np.float64))**2)>0.05

#ecfp6Var005Recon=ecfp6Mat[:,featSelVar005].copy()
#sampleSelVar005=ecfp6Var005Recon.getnnz(1)>=0.5
#ecfp6Var005Recon=ecfp6Var005Recon[sampleSelVar005]
#ecfp6Var005Recon.eliminate_zeros()
#ecfp6Var005Recon.tocsr()
#ecfp6Var005Recon.sort_indices()

#assert(np.all(ecfp6Var005Recon.data==ecfp6Var005Mat.data))
#assert(np.all(ecfp6Var005Recon.indices==ecfp6Var005Mat.indices))
#assert(np.all(ecfp6Var005Recon.indptr==ecfp6Var005Mat.indptr))
#assert(np.all(sampleECFP6Ind.index.values[sampleSelVar005]==sampleECFP6Var005Ind.index.values))
#assert(np.all(featureECFP6Ind.index.values[featSelVar005]==featureECFP6Var005Ind.index.values))

