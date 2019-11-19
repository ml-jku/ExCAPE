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



#python3 $HOME/mysources/pipeline2/excapedb/mystep1Norm.py -maxProc 10 -availableGPUs -sizeFact 1.0 -dataset ecfp6 -ofolds 0 1 2 -ifolds 0 1 2 -pStart 0 -pEnd 32 -continueComputations -saveComputations -startMark start -finMark finished -epochs 300
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
parser.add_argument("-maxProc", help="Max. Nr. of Processes", type=int, default=3)
parser.add_argument("-availableGPUs", help="Available GPUs", nargs='*', type=int, default=[0])
parser.add_argument("-sizeFact", help="Size Factor GPU Scheduling", type=float, default=1.0)
parser.add_argument("-originalData", help="Path for original data", type=str, default=os.getenv("HOME")+"/mydatasets/excapedb/current/")
parser.add_argument("-dataset", help="Dataset Name", type=str, default="ecfp6")
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=os.getenv("HOME")+"/mydatasets/excapedb/res/")
parser.add_argument("-ofolds", help="Outer Folds", nargs='+', type=int, default=[0,1,2])
parser.add_argument("-ifolds", help="Inner Folds", nargs='+', type=int, default=[0,1,2])
parser.add_argument("-pStart", help="Parameter Start Index", type=int, default=0)
parser.add_argument("-pEnd", help="Parameter End Index", type=int, default=float('inf'))
parser.add_argument("-continueComputations", help="continueComputations", action='store_true')
parser.add_argument("-saveComputations", help="saveComputations", action='store_true')
parser.add_argument("-startMark", help="startMark", type=str, default="start")
parser.add_argument("-finMark", help="finMark", type=str, default="finished")
parser.add_argument("-epochs", help="Nr. Epochs", type=int, default=300)
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
compInnerFolds=args.ifolds
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
batchSize=args.batchSize
stopAtMinibatch=np.nan



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



if len(availableGPUs)>0.5:
  if not os.path.exists(savePath+"hyperSize"+hyperparamSetName+".npy"):
    hyperSize=np.zeros(len(hyperParams))
    totalSize=np.Inf
  else:
    hyperSize=np.fromfile(savePath+"hyperSize"+hyperparamSetName+".npy", dtype=np.int64)
    totalSize=np.fromfile(savePath+"totalSize"+hyperparamSetName+".npy", dtype=np.int64)
  gpuInitNr=0
  gpuAllocArray=np.repeat(totalSize, len(availableGPUs))
  processAllocArray=dict()
runningProc=list()



for outerFold in compOuterFolds:
  for innerFold in compInnerFolds:
    if innerFold==outerFold:
      continue
    
    if outerFold<0:
      trainSamples=sorted(set(allSamples)-set(folds[innerFold]))
      testSamples=sorted(folds[innerFold])
    else:
      trainSamples=sorted(set(allSamples)-set(folds[innerFold]+folds[outerFold]))
      testSamples=sorted(folds[innerFold])
    exec(open(basePath+'excapedb/prepareDatasetsLocal.py').read(), globals())
    
    
    
    for paramNr in compParams:
      savePrefix0="step1_o"+'{0:04d}'.format(outerFold+1)+"_i"+'{0:04d}'.format(innerFold+1)+"_p"+'{0:04d}'.format(hyperParams.index.values[paramNr])
      savePrefix=savePath+savePrefix0
      if os.path.isfile(savePrefix+"."+finMark+".pckl") and (not continueComputations):
        continue
      saveFilename=savePrefix+"."+startMark+".pckl"
      if os.path.isfile(saveFilename):
        continue
      
      
      
      if len(availableGPUs)>0.5:
        gpuInitNr=gpuInitNr+1
        initGPUDeviceAlloc=gpuInitNr%len(availableGPUs)
        usedGPUMemoryAlloc=int(hyperSize[hyperParams.index.values[paramNr]]*sizeFact)
        usedGPUDeviceAlloc=initGPUDeviceAlloc
        while True:
          if gpuAllocArray[usedGPUDeviceAlloc]-usedGPUMemoryAlloc>0:
            break
          usedGPUDeviceAlloc=(usedGPUDeviceAlloc+1)%len(availableGPUs)
          if usedGPUDeviceAlloc==initGPUDeviceAlloc:
            time.sleep(1)
            for entryNr in list(range(len(runningProc)-1, -1, -1)):
              if(os.waitpid(runningProc[entryNr], os.WNOHANG)!=(0,0)):
                del runningProc[entryNr]
            for pid in np.setdiff1d(list(processAllocArray.keys()), runningProc):
              usedGPUDeviceFree=processAllocArray[pid][0]
              usedGPUMemoryFree=processAllocArray[pid][1]
              del processAllocArray[pid]
              gpuAllocArray[usedGPUDeviceFree]=gpuAllocArray[usedGPUDeviceFree]+usedGPUMemoryFree
        gpuAllocArray[usedGPUDeviceAlloc]=gpuAllocArray[usedGPUDeviceAlloc]-usedGPUMemoryAlloc
        print(gpuAllocArray[usedGPUDeviceAlloc])
      
      
      
      while(len(runningProc)>=maxProcesses):
        time.sleep(1)
        for entryNr in list(range(len(runningProc)-1, -1, -1)):
          if(os.waitpid(runningProc[entryNr], os.WNOHANG)!=(0,0)):
            del runningProc[entryNr]
      
      if maxProcesses>1.5:
        forkRET=os.fork()
        if forkRET!=0:
          runningProc.append(forkRET)
          if len(availableGPUs)>0.5:
            processAllocArray[forkRET]=(usedGPUDeviceAlloc, usedGPUMemoryAlloc)
          continue
      
      
      
      if len(availableGPUs)>0.5:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(availableGPUs[usedGPUDeviceAlloc])
    
    
    
      saveFile=open(saveFilename, "wb")
      startNr=0
      pickle.dump(startNr, saveFile)
      saveFile.close()
      dbgOutput=open(dbgPath+savePrefix0+".dbg", "w")
      print(hyperParams.iloc[paramNr], file=dbgOutput)
      
      
      
      basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture
      
      modelScript=basePath+'model'+basicArchitecture[0]+'.py'
      loadScript=basePath+'step1Load.py'
      saveScript=basePath+'step1Save.py'
      runEpochs=True
      exec(open(basePath+'runEpochs'+basicArchitecture[0]+'.py').read(), globals())
      
      
      
      dbgOutput.close()
      
      
      
      if maxProcesses>1.5:
        if forkRET==0:
          os._exit(0)



while(len(runningProc)>0.5):
  time.sleep(1)
  for entryNr in list(range(len(runningProc)-1, -1, -1)):
    if(os.waitpid(runningProc[entryNr], os.WNOHANG)!=(0,0)):
      del runningProc[entryNr]
  if len(availableGPUs)>0.5:
    for pid in np.setdiff1d(list(processAllocArray.keys()), runningProc):
      usedGPUDeviceFree=processAllocArray[pid][0]
      usedGPUMemoryFree=processAllocArray[pid][1]
      del processAllocArray[pid]
      gpuAllocArray[usedGPUDeviceFree]=gpuAllocArray[usedGPUDeviceFree]+usedGPUMemoryFree
