#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle


f=open(dataPathSave+'folds'+str(mySampleIndex)+'.pckl', "rb")
folds=pickle.load(f)
f.close()

#f=open(dataPathSave+'labelsHard.pckl', "rb")
#targetMat=pickle.load(f)
#sampleAnnInd=pickle.load(f)
#targetAnnInd=pickle.load(f)
#f.close()

#if myAffinityLevel in [5, 6, 7, 8]:
#  targetMat=targetMat[:,np.array([x.startswith('t'+str(myAffinityLevel-4)+'_trg') for x in targetAnnInd.index.values.tolist()])]
#  targetMat=targetMat.copy().tocsr()
#  targetMat.sort_indices()
#  targetAnnInd=targetAnnInd[np.array([x.startswith('t'+str(myAffinityLevel-4)+'_trg') for x in targetAnnInd.index.values.tolist()])]
#  targetAnnInd=targetAnnInd-targetAnnInd.min()

f=open(dataPathSave+'labelsExCAPE.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
targetMatTransposed=targetMat[sampleAnnInd[list(itertools.chain(*folds))]].T.tocsr()
targetMatTransposed.sort_indices()
trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
#targetMat=targetMat[:, trainPosOverall+trainNegOverall>100].copy()
#targetMat.sort_indices()

denseOutputData=targetMat.A
sparseOutputData=targetMat



if datasetName=="ecfp6":
  f=open(dataPathSave+'ecfp6.pckl', "rb")
  ecfp6Mat=pickle.load(f)
  sampleECFP6Ind=pickle.load(f)
  featureECFP6Ind=pickle.load(f)
  f.close()
  
  if applyTanhToSparse:
    ecfp6Mat=np.tanh(ecfp6Mat)
  
  denseInputData=None
  denseSampleIndex=None
  sparseInputData=ecfp6Mat
  sparseSampleIndex=sampleECFP6Ind
  
  sparsenesThr=0.0025
elif datasetName=="ecfp6Folded":
  f=open(dataPathSave+'ecfp6Folded.pckl', "rb")
  ecfp6FoldedMat=pickle.load(f)
  sampleECFP6FoldedInd=pickle.load(f)
  featureECFP6FoldedInd=pickle.load(f)
  f.close()
  
  denseInputData=ecfp6FoldedMat
  denseSampleIndex=sampleECFP6FoldedInd
  sparseInputData=None
  sparseSampleIndex=None
elif datasetName=="chem2vec":
  f=open(dataPathSave+'chem2vec.pckl', "rb")
  chem2vecMat=pickle.load(f)
  sampleChem2VecInd=pickle.load(f)
  featureChem2VecInd=pickle.load(f)
  f.close()

  denseInputData=chem2vecMat
  denseSampleIndex=sampleChem2VecInd
  sparseInputData=None
  sparseSampleIndex=None
elif datasetName=="ecfp6Var005":
  f=open(dataPathSave+'ecfp6Var005.pckl', "rb")
  ecfp6Var005Mat=pickle.load(f)
  sampleECFP6Var005Ind=pickle.load(f)
  featureECFP6Var005Ind=pickle.load(f)
  f.close()
  
  if applyTanhToSparse:
    ecfp6Var005Mat=np.tanh(ecfp6Var005Mat)
  
  denseInputData=None
  denseSampleIndex=None
  sparseInputData=ecfp6Var005Mat
  sparseSampleIndex=sampleECFP6Var005Ind
  
  sparsenesThr=-1.0 #0.005 #possibly change to 0.001, if expected that more features lead to better performance


allSamples=np.array([], dtype=np.int64)
if not (denseInputData is None):
  allSamples=np.union1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.union1d(allSamples, sparseSampleIndex.index.values)
allSamples=np.union1d(allSamples, sampleAnnInd.index.values)
  
if not (denseInputData is None):
  allSamples=np.intersect1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.intersect1d(allSamples, sparseSampleIndex.index.values)
allSamples=np.intersect1d(allSamples, sampleAnnInd.index.values)
allSamples=allSamples.tolist()



if not (denseInputData is None):
  folds=[np.intersect1d(fold, denseSampleIndex.index.values).tolist() for fold in folds]
if not (sparseInputData is None):
  folds=[np.intersect1d(fold, sparseSampleIndex.index.values).tolist() for fold in folds]