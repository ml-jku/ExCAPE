#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

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
import os
import sys
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import time
import gc
import argparse
basePath=os.getenv("HOME")+"/mysources/pipeline2/"
projectName="excapedb"
projectPath=basePath+projectName+"/"
utilsLib=imp.load_source(basePath+'utilsLib.py', basePath+"utilsLib.py")

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format



#python3 $HOME/mysources/pipeline2/excapedb/mystep2BNorm.py -maxProc 10 -availableGPUs -sizeFact 1.0 -dataset ecfp6 -ofolds 0 1 2 -pStart 0 -pEnd 32 -continueComputations -saveComputations -startMark start -finMark finished -epochs 300 -mode simple
parser = argparse.ArgumentParser()
parser.add_argument("-batchSize", help="ExCAPE Batch size", type=int, default=128)
parser.add_argument("-sampleIndex", help="ExCAPE SampleIndex", type=int, default=0)
parser.add_argument("-affinityLevel", help="ExCAPE Affinity Level", type=int, default=6)
#AffinityLevel 0: anything except 5, 6, 7, 8
#AffinityLevel 5: ExCAPE Affinity Level 5
#AffinityLevel 6: ExCAPE Affinity Level 6
#AffinityLevel 7: ExCAPE Affinity Level 7
#AffinityLevel 8: ExCAPE Affinity Level 8
parser.add_argument("-hyperparam", help="Hyperparameter set name ", type=str, default="")
parser.add_argument("-maxProc", help="Max. Nr. of Processes", type=int, default=10)
parser.add_argument("-availableGPUs", help="Available GPUs", nargs='*', type=int, default=[0])
parser.add_argument("-sizeFact", help="Size Factor GPU Scheduling", type=float, default=1.0)
parser.add_argument("-originalData", help="Path for original data", type=str, default=os.getenv("HOME")+"/mydatasets/excapedb/current/")
parser.add_argument("-dataset", help="Dataset Name", type=str, default="ecfp6")
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=os.getenv("HOME")+"/mydatasets/excapedb/res/")
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0])
parser.add_argument("-pStart", help="Parameter Start Index", type=int, default=0)
parser.add_argument("-pEnd", help="Parameter End Index", type=int, default=float('inf'))
parser.add_argument("-continueComputations", help="continueComputations", action='store_true')
parser.add_argument("-saveComputations", help="saveComputations", action='store_true')
parser.add_argument("-startMark", help="startMark", type=str, default="start")
parser.add_argument("-finMark", help="finMark", type=str, default="finished")
parser.add_argument("-epochs", help="Nr. Epochs", type=int, default=300)
parser.add_argument("-sameMinibatches", help="same nr. of epochs (default) or minibatches", action='store_true')
parser.add_argument("-mode", help="Hyperparam selection level", choices=['simple', 'target', 'targetAndEpoch'], default="simple")
args = parser.parse_args()



mySampleIndex=args.sampleIndex
myAffinityLevel=args.affinityLevel



hyperparamSetName=args.hyperparam
exec(open(projectPath+'hyperparams'+hyperparamSetName+'.py').read(), globals())



maxProcesses=args.maxProc
availableGPUs=args.availableGPUs
sizeFact=args.sizeFact

dataPathSave=args.originalData

datasetName=args.dataset
saveBasePath=args.saveBasePath
if not os.path.exists(saveBasePath):
  os.makedirs(saveBasePath)
savePath=saveBasePath+datasetName+"/"
if not os.path.exists(savePath):
  os.makedirs(savePath)  
dbgPath=savePath+"dbg/"
if not os.path.exists(dbgPath):
  os.makedirs(dbgPath)

compOuterFolds=args.ofolds
paramStart=args.pStart
paramEnd=min(hyperParams.shape[0], args.pEnd)
compParams=list(range(paramStart, paramEnd))

if not bool(getattr(sys, 'ps1', sys.flags.interactive)):
  continueComputations=args.continueComputations
  saveComputations=args.saveComputations
else:
  continueComputations=True
  saveComputations=True
startMark=args.startMark
finMark=args.finMark

nrEpochs=args.epochs
sameEpochs=not args.sameMinibatches
mode=args.mode
batchSize=args.batchSize



applyTanhToSparse=False
exec(open(projectPath+'loadData.py').read(), globals())

normalizeGlobalDense=False
normalizeGlobalSparse=False
normalizeLocalDense=False
normalizeLocalSparse=False
if not denseInputData is None:
  normalizeLocalDense=True
if not sparseInputData is None:
  normalizeLocalDense=True
  normalizeLocalSparse=True
exec(open(basePath+'prepareDatasetsGlobal.py').read(), globals())



minibatchesPerReportTrain=int(int(np.mean([len(x) for x in folds]))/batchSize)*20
minibatchesPerReportTest=int(int(np.mean([len(x) for x in folds]))/batchSize)



useDenseOutputNetTrain=False
useDenseOutputNetPred=False
computeTrainPredictions=True
compPerformanceTrain=True
computeTestPredictions=True
compPerformanceTest=True

logPerformanceAtBestIter=False
savePredictionsAtBestIter=False



for outerFold in compOuterFolds:
  
  
  
  takeMinibatch=[]
  for evalFold in range(0, len(folds)):
    if evalFold==outerFold:
      continue
    if outerFold<0:
      compNrMinibatches=float(nrEpochs)*math.ceil(float(len(list(set(allSamples)-set(folds[evalFold]))))/float(batchSize))
    else:
      compNrMinibatches=float(nrEpochs)*math.ceil(float(len(list(set(allSamples)-set(folds[evalFold]+folds[outerFold]))))/float(batchSize))
    compLastMinibatch=math.trunc(compNrMinibatches/minibatchesPerReportTest)-1
    takeMinibatch.append(compLastMinibatch)
  
  stopAtMinibatch=np.nan
  if not sameEpochs:
    stopAtMinibatch=min(takeMinibatch)
    takeMinibatch=[stopAtMinibatch for tm in takeMinibatch]
  
  if (not sameEpochs) or (not (mode=='simple' or mode=='target')):
    perfFiles=[]
    for evalFold in range(0, len(folds)):
      if evalFold==outerFold:
        continue
      perfFiles.append([])
      for paramNr in range(0, hyperParams.shape[0]):
        perfFiles[-1].append(savePath+"step1_o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(evalFold+1)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])+".test.aucMB.pckl")
    innerAUC, innerAUCExt=utilsLib.getInnerPerformance(perfFiles, takeMinibatch)
    aucMeanSmoothExt=np.nan_to_num(np.array([pd.stats.moments.rolling_mean(innerAUCExt[ind], center=True, window=2) for ind in range(0, innerAUCExt.shape[0])]))
    bestParamIndPerTargetAndTimestep=np.array([np.unravel_index(np.argmax(aucMeanSmoothExt[:,:,ind]), aucMeanSmoothExt.shape[0:2]) for ind in range(aucMeanSmoothExt.shape[2])])
    bestParamIndPerTarget=innerAUC.argmax(axis=0)
    bestParamIndOverall=np.nanmean(innerAUC, axis=1).argmax()
  
  if sameEpochs and (mode=='simple' or mode=='target'):
    perfFiles=[]
    for evalFold in range(0, len(folds)):
      if evalFold==outerFold:
        continue
      perfFiles.append([])
      for paramNr in range(0, hyperParams.shape[0]):
        perfFiles[-1].append(savePath+"step1_o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(evalFold+1)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])+".test.aucEnd.pckl")
    innerAUC=utilsLib.getInnerEndPerformance(perfFiles)
    bestParamIndPerTarget=innerAUC.argmax(axis=0)
    bestParamIndOverall=np.nanmean(innerAUC, axis=1).argmax()
  
  
  
  if mode=='simple':
    mycompParams=[bestParamIndOverall]
    bestParamInd=np.array([bestParamIndOverall]*nrOutputTargets)
    savePredictionsAtBestIter=False
    logPerformanceAtBestIter=False
  elif mode=='target':
    #mycompParams=np.sort(np.unique(bestParamIndPerTarget))
    mycompParams=compParams
    bestParamInd=bestParamIndPerTarget
    savePredictionsAtBestIter=False
    logPerformanceAtBestIter=False
  elif mode=='targetAndEpoch':
    #mycompParams=np.sort(np.unique(bestParamIndPerTargetAndTimestep[:,0]))
    mycompParams=compParams
    bestParamInd=bestParamIndPerTargetAndTimestep[:,0]
    bestIterPerTask=bestParamIndPerTargetAndTimestep[:,1]
    savePredictionsAtBestIter=True
    logPerformanceAtBestIter=True
  
  
  
  if outerFold<0:
    trainSamples=sorted(set(allSamples))
    testSamples=sorted(set(allSamples))
    
    useDenseOutputNetPred=True
    compPerformanceTest=False
    savePredictionsAtBestIter=savePredictionsAtBestIter
    logPerformanceAtBestIter=False
    savePredictions=True
    logPerformance=False
  else:
    trainSamples=sorted(set(allSamples)-set(folds[outerFold]))
    testSamples=sorted(folds[outerFold])
    
    useDenseOutputNetPred=True
    compPerformanceTest=True
    savePredictionsAtBestIter=savePredictionsAtBestIter
    logPerformanceAtBestIter=logPerformanceAtBestIter
    savePredictions=True
    logPerformance=True
  exec(open(basePath+'excapedb/prepareDatasetsLocal.py').read(), globals())
  
  
  
  if useDenseOutputNetPred:
    predDenseTestSummary=np.zeros((len(testSamples), nrOutputTargets))
    predDenseTestSummary[:]=np.nan
    if compPerformanceTest:
      sumTestAUCSummary=np.zeros(nrOutputTargets)
      sumTestAUCSummary[:]=np.nan
      sumTestAPSummary=np.zeros(nrOutputTargets)
      sumTestAPSummary[:]=np.nan
      #reportAUCBestIter=np.zeros(nrOutputTargets)
      #reportAPBestIter=np.zeros(nrOutputTargets)
  
  if not useDenseOutputNetPred:
    predSparseTestSummary=testSparseOutput.copy().astype(np.float32)
    predSparseTestSummary.data[:]=-1
    if compPerformanceTest:
      sumTestAUCSummary=np.zeros(nrOutputTargets)
      sumTestAUCSummary[:]=np.nan
      sumTestAPSummary=np.zeros(nrOutputTargets)
      sumTestAPSummary[:]=np.nan
      #reportAUCBestIter=np.zeros(nrOutputTargets)
      #reportAPBestIter=np.zeros(nrOutputTargets)  
  
  
  
  for paramNr in mycompParams:
    savePrefix0="step2_o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(0)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])
    #savePrefix0="step1_o"+'{0:04d}'.format(0)+"_i"+'{0:04d}'.format(outerFold+1)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])
    savePrefix=savePath+savePrefix0
    
    
    
    usedGPUDeviceAlloc=0
    if len(availableGPUs)>0.5:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(availableGPUs[usedGPUDeviceAlloc])
    
    
    
    dbgOutput=open(dbgPath+savePrefix0+".dbgB", "w")
    print(hyperParams.iloc[paramNr], file=dbgOutput)
    
    """
    outerFold=1
    paramNr=0
    savePrefix0="test_step2_o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(0)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])
    savePrefix=savePath+savePrefix0
    dbgOutput=sys.stdout
    """
    
    
    
    if mode=='simple':
      sumTestAUC=None
      sumTestAP=None
      predDenseTest=None
      predSparseTest=None

      basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture
      
      modelScript=basePath+'model'+basicArchitecture[0]+'.py'
      loadScript=basePath+'step2Load.py'
      saveScript=''
      runEpochs=False
      exec(open(basePath+'runEpochs'+basicArchitecture[0]+'.py').read(), globals())
      
      if startEpoch!=nrEpochs-1:
        raise Exception('Training of step2 not finished so far!!!') 
      if useDenseOutputNetPred:
        predDenseTestSummary[:, bestParamInd==paramNr]=predDenseTest[:, bestParamInd==paramNr]
        if savePredictionsAtBestIter:
          pass
        if compPerformanceTest:
          sumTestAUCSummary[bestParamInd==paramNr]=sumTestAUC[bestParamInd==paramNr]
          sumTestAPSummary[bestParamInd==paramNr]=sumTestAP[bestParamInd==paramNr]
      
      if not useDenseOutputNetPred:
        nonz=predSparseTest.nonzero()
        sel=np.in1d(nonz[1], np.where(bestParamInd==paramNr)[0])
        if(np.any(sel)):
          predSparseTestSummary[nonz[0][sel], nonz[1][sel]]=predSparseTest[nonz[0][sel], nonz[1][sel]]
        if savePredictionsAtBestIter:
          pass
        if compPerformanceTest:
          sumTestAUCSummary[bestParamInd==paramNr]=sumTestAUC[bestParamInd==paramNr]
          sumTestAPSummary[bestParamInd==paramNr]=sumTestAP[bestParamInd==paramNr]
      
    elif mode=='target':
      sumTestAUC=None
      sumTestAP=None
      predDenseTest=None
      predSparseTest=None

      basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture
      
      modelScript=basePath+'model'+basicArchitecture[0]+'.py'
      loadScript=basePath+'step2Load.py'
      saveScript=''
      runEpochs=False
      exec(open(basePath+'runEpochs'+basicArchitecture[0]+'.py').read(), globals())
      
      if startEpoch!=nrEpochs-1:
        raise Exception('Training of step2 not finished so far!!!') 
      if useDenseOutputNetPred:
        predDenseTestSummary[:, bestParamInd==paramNr]=predDenseTest[:, bestParamInd==paramNr]
        if savePredictionsAtBestIter:
          pass
        if compPerformanceTest:
          sumTestAUCSummary[bestParamInd==paramNr]=sumTestAUC[bestParamInd==paramNr]
          sumTestAPSummary[bestParamInd==paramNr]=sumTestAP[bestParamInd==paramNr]
      
      if not useDenseOutputNetPred:
        nonz=predSparseTest.nonzero()
        sel=np.in1d(nonz[1], np.where(bestParamInd==paramNr)[0])
        if(np.any(sel)):
          predSparseTestSummary[nonz[0][sel], nonz[1][sel]]=predSparseTest[nonz[0][sel], nonz[1][sel]]
        if savePredictionsAtBestIter:
          pass
        if compPerformanceTest:
          sumTestAUCSummary[bestParamInd==paramNr]=sumTestAUC[bestParamInd==paramNr]
          sumTestAPSummary[bestParamInd==paramNr]=sumTestAP[bestParamInd==paramNr]
      
    elif mode=='targetAndEpoch':
      loadScript=basePath+'step2Load.py'
      exec(open(loadScript).read(), globals())
      if logPerformanceAtBestIter:
        saveFilename=savePrefix+".test.aucBI.npy"
        if os.path.isfile(saveFilename):
          reportAUCBestIter=np.load(saveFilename)
        
        saveFilename=savePrefix+".test.apBI.npy"
        if os.path.isfile(saveFilename):
          reportAPBestIter=np.load(saveFilename)  
      
      
      
      if savePredictionsAtBestIter:
        if useDenseOutputNetPred:
          saveFilename=savePrefix+".evalPredict.pckl"
          if os.path.isfile(saveFilename):
            saveFile=open(saveFilename, "rb")
            predDenseBestIter=pickle.load(saveFile)
            saveFile.close()
        else:
          saveFilename=savePrefix+".evalPredict.pckl"
          if os.path.isfile(saveFilename):
            saveFile=open(saveFilename, "rb")
            predSparseBestIter=pickle.load(saveFile)
            saveFile.close()
      
      
      
      if useDenseOutputNetPred:
        if savePredictionsAtBestIter:
          predDenseTestSummary[:,bestParamInd==paramNr]=predDenseBestIter[:,bestParamInd==paramNr]
        if compPerformanceTest:
          sumTestAUCSummary[bestParamInd==paramNr]=reportAUCBestIter[bestParamInd==paramNr]
          sumTestAPSummary[bestParamInd==paramNr]=reportAPBestIter[bestParamInd==paramNr]
          #sumTestAUC=np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseBestIter))
          #sumTestAP=np.array(utilsLib.calculateAPs(testDenseOutput, predDenseBestIter))

      if not useDenseOutputNetPred:
        if savePredictionsAtBestIter:
          nonz=predSparseBestIter.nonzero()
          sel=np.in1d(nonz[1], np.where(bestParamInd==paramNr)[0])
          if(np.any(sel)):
            predSparseTestSummary[nonz[0][sel], nonz[1][sel]]=predSparseBestIter[nonz[0][sel], nonz[1][sel]]
          predSparseBestIterTransposed=predSparseBestIter.copy().T.tocsr()
        if compPerformanceTest:
          sumTestAUCSummary[bestParamInd==paramNr]=reportAUCBestIter[bestParamInd==paramNr]
          sumTestAPSummary[bestParamInd==paramNr]=reportAPBestIter[bestParamInd==paramNr]
          #sumTestAUC=np.array(utilsLib.calculateSparseAUCs(testSparseOutputTransposed, predSparseBestIterTransposed))
          #sumTestAP=np.array(utilsLib.calculateSparseAPs(testSparseOutputTransposed, predSparseBestIterTransposed))
  
  
  
  dbgOutput.close()
  
  
  
  savePrefix0="step2_o"+'{0:04d}'.format(outerFold+1)
  savePrefix=savePath+savePrefix0
  
  
  
  if logPerformance:
    saveFilename=savePrefix+".test.aucSum.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(sumTestAUCSummary, saveFile)
    saveFile.close()
    
    saveFilename=savePrefix+".test.apSum.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(sumTestAPSummary, saveFile)
    saveFile.close()
  
  if savePredictions:
    if useDenseOutputNetPred:
      saveFilename=savePrefix+".test.predSum.pckl"
      saveFile=open(saveFilename, "wb")
      pickle.dump(predDenseTestSummary.astype(np.float32), saveFile)
      saveFile.close()
    else:
      saveFilename=savePrefix+".test.predSum.pckl"
      saveFile=open(saveFilename, "wb")
      pickle.dump(predSparseTestSummary, saveFile)
      saveFile.close()
    
    #if useDenseOutputNetPred:
    #  saveFilename=savePrefix+".test.true.pckl"
    #  saveFile=open(saveFilename, "wb")
    #  pickle.dump(testDenseOutput.astype(np.float32), saveFile)
    #  saveFile.close()
    #else:
    #  saveFilename=savePrefix+".test.true.pckl"
    #  saveFile=open(saveFilename, "wb")
    #  pickle.dump(testSparseOutput, saveFile)
    #  saveFile.close()
    
    saveFilename=savePrefix+".test.cmpNames.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(np.array(testSamples), saveFile)
    saveFile.close()
    
    saveFilename=savePrefix+".test.targetNames.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(np.array(targetAnnInd.index.values), saveFile)
    saveFile.close()
  
  sumTestAUCSummary=None
  sumTestAPSummary=None
  predDenseTestSummary=None
  predSparseTestSummary=None
  

    
    
