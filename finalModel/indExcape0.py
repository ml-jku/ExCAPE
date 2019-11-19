#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

#module load Tensorflow/1.12.0-GCC-6.3.0-2.27-Python-3.6.1

import os
generalDir="/scratch/work/project/excape-public/linzDLModel"



from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import math
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import sklearn
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.metrics
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import imp
import sys
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import time
import gc
import argparse
basePath=generalDir+"/sources/"
utilsLib=imp.load_source(basePath+'utilsLib.py', basePath+"utilsLib.py")

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format

availableGPUs=[]
if len(availableGPUs)>0.5:
  os.environ['CUDA_VISIBLE_DEVICES']=str(availableGPUs[0])





dictionary0 = {
  'basicArchitecture': [('ReLUScalingDropout', 'reluScaling')],
  'l2Penalty': [0.0],
  'learningRate': [0.01],
  'l1Penalty': [0.0],
  'layerForm': ["rect"],
  'idropout': [0.2],
  'dropout': [0.5],
  'nrStart': [4096],
  'nrLayers': [3],
  'mom': [0.0]
}

hyperParams=pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys()) 





dataPathSave=generalDir+"/current/"
datasetName='ecfp6'
saveBasePath=generalDir+"/model/"
savePath=saveBasePath+datasetName+"/"



nrSparseFeatures=0

featureFile=open(dataPathSave+'featureData.pckl', "rb")
nrDenseFeatures=pickle.load(featureFile)
selectedFeatures=pickle.load(featureFile)
trainDenseMean1=pickle.load(featureFile)
trainDenseStd1=pickle.load(featureFile)
trainDenseMean2=pickle.load(featureFile)
trainDenseStd2=pickle.load(featureFile)
featureFile.close()

targetNamesFile=open(dataPathSave+'targetNames.pckl', "rb")
myTargetNames=pickle.load(targetNamesFile)
featureFile.close()









ecfp6=pd.read_csv("/scratch/work/project/excape-public/data_release_v5/version5/ecfp6_counts.txt.gz", sep="\t", header=0, index_col=None).drop_duplicates()
sampleECFP6Ind=pd.Series(data=np.arange(len(np.unique(ecfp6.iloc[:,0].values))), index=np.unique(ecfp6.iloc[:,0].values))
featureECFP6Ind=pd.Series(data=np.arange(len(np.unique(ecfp6.iloc[:,1].values))), index=np.unique(ecfp6.iloc[:,1].values))
ecfp6Mat=scipy.sparse.coo_matrix((ecfp6.iloc[:,2], (sampleECFP6Ind[ecfp6.iloc[:,0]], featureECFP6Ind[ecfp6.iloc[:,1]])), shape=(sampleECFP6Ind.max()+1, featureECFP6Ind.max()+1))
ecfp6Mat=ecfp6Mat.tocsr()
ecfp6Mat.sort_indices()

ind_ecfp6Mat=ecfp6Mat
ind_sampleECFP6Ind=sampleECFP6Ind
ind_featureECFP6Ind=featureECFP6Ind
missingFeatures=np.setdiff1d(selectedFeatures, ind_featureECFP6Ind.index.values)
if(len(missingFeatures)>0.5):
  zeroMat=scipy.sparse.csr_matrix((len(ind_sampleECFP6Ind.index.values), len(missingFeatures)))
  ind_ecfp6Mat=scipy.sparse.hstack([ind_ecfp6Mat, zeroMat]).tocsr()
  ind_ecfp6Mat.sort_indices()
  ind_featureECFP6Ind=pd.Series(index=np.array(ind_featureECFP6Ind.index.values.tolist()+missingFeatures.tolist()), data=np.array(ind_featureECFP6Ind.values.tolist()+np.arange(len(ind_featureECFP6Ind.values), len(ind_featureECFP6Ind.values)+len(missingFeatures)).tolist()))

testDenseInput=ind_ecfp6Mat[:, ind_featureECFP6Ind[selectedFeatures].values].tocsr().A
testDenseInput=(testDenseInput-trainDenseMean1)/trainDenseStd1
testDenseInput=np.tanh(testDenseInput)
testDenseInput=(testDenseInput-trainDenseMean2)/trainDenseStd2
testDenseInput=np.nan_to_num(testDenseInput)






sparseOutputData=None
#minibatchesPerReportTrain=49740
#minibatchesPerReportTest=2487
batchSize=128
nrOutputTargets=526
trainBias=np.zeros(nrOutputTargets)
stopAtMinibatch=np.nan
dbgOutput=sys.stdout
normalizeGlobalSparse=False
normalizeLocalSparse=False
continueComputations=True
saveComputations=False
nrEpochs=0
computeTrainPredictions=False
compPerformanceTrain=False
computeTestPredictions=True
compPerformanceTest=False
useDenseOutputNetPred=True
savePredictionsAtBestIter=False
logPerformanceAtBestIter=False
runEpochs=False


outerFold=-1
paramNr=0
savePrefix0="step2_o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(0)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])
savePrefix=savePath+savePrefix0
basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture

modelScript=basePath+'model'+basicArchitecture[0]+'.py'
loadScript=basePath+'step2Load.py'
saveScript=""

testSamples=ind_sampleECFP6Ind.index.values
exec(open(basePath+'runEpochs'+basicArchitecture[0]+'.py').read(), globals())

predMatrix=pd.DataFrame(data=predDenseTest, index=testSamples, columns=myTargetNames)









#---------------------------------------------------------------------------------------
#only to be used as a quick check for the Excape data (== version5/ecfp6_counts.txt.gz), whether loading the model, etc. was fine

f=open(dataPathSave+'labelsExCAPE.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()
testDenseOutput=(targetMat.A[sampleAnnInd[predMatrix.index.values].values])[:, targetAnnInd[predMatrix.columns.values]].copy()
print(np.nanmean(np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseTest))))
testDenseOutput=targetMat.A.copy()
print(np.nanmean(np.array(utilsLib.calculateAUCs(testDenseOutput, predMatrix.loc[sampleAnnInd.index.values].loc[:,targetAnnInd.index.values].values))))
#both prints should give a value of around 0.98 (training AUC)





