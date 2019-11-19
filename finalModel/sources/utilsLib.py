#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

from __future__ import absolute_import, division, print_function
import numpy as np
import sklearn
import sklearn.metrics
import pickle
import pandas as pd

def calculateAUCs(t, p):
  aucs = []
  for i in range(p.shape[1]):
    targ = t[:, i] > 0.5
    pred = p[:, i]
    idx = np.abs(t[:, i]) > 0.5
    try:
      aucs.append(sklearn.metrics.roc_auc_score(targ[idx], pred[idx]))
    except ValueError:
      aucs.append(np.nan)
  return aucs

def calculateSparseAUCs(t, p):
  aucs = []
  for i in range(p.shape[0]):
    targ = t[i].data > 0.5
    pred = p[i].data
    try:
      aucs.append(sklearn.metrics.roc_auc_score(targ, pred))
    except ValueError:
      aucs.append(np.nan)
  return aucs
  
def calculateAPs(t, p):
  aucs = []
  for i in range(p.shape[1]):
    targ = t[:, i] > 0.5
    pred = p[:, i]
    idx = np.abs(t[:, i]) > 0.5
    try:
      precision, recall, thresholds=sklearn.metrics.precision_recall_curve(targ[idx], pred[idx])
      area=sklearn.metrics.auc(recall, precision)
      aucs.append(area)
      #aucs.append(sklearn.metrics.average_precision_score(targ[idx], pred[idx]))
    except ValueError:
      aucs.append(np.nan)
  return aucs

def calculateSparseAPs(t, p):
  aucs = []
  for i in range(p.shape[0]):
    targ = t[i].data > 0.5
    pred = p[i].data
    try:
      precision, recall, thresholds=sklearn.metrics.precision_recall_curve(targ, pred)
      area=sklearn.metrics.auc(recall, precision)
      aucs.append(area)
      #aucs.append(sklearn.metrics.average_precision_score(targ, pred))
    except ValueError:
      aucs.append(np.nan)
  return aucs
  
def bestSettings(perfFiles, nrParams):
  aucFold=[]
  for foldInd in range(0, len(perfFiles)):
    innerFold=-1
    aucParam=[]
    for paramNr in range(0, nrParams):
      #try:
      saveFile=open(perfFiles[foldInd][paramNr], "rb")
      aucRun=pickle.load(saveFile)
      saveFile.close()
      #except:
      #  pass
      aucRun=np.array(aucRun)
      if(len(aucRun)>0):
        aucParam.append(aucRun)
    
    maxShape0=max([x.shape[0] for x in aucParam])
    shape1=aucParam[0].shape[1]
    aucParam2=np.zeros(shape=(len(aucParam), maxShape0, shape1))
    aucParam2[:,:,:]=np.nan
    for paramNr in range(0, nrParams):
      aucParam2[paramNr,0:aucParam[paramNr].shape[0],:]=aucParam[paramNr]
    aucParam=aucParam2
    
    if(len(aucParam)>0):
      aucFold.append(aucParam)
  aucFoldPad=np.zeros(shape=np.concatenate([np.array([len(aucFold)], dtype=np.int64), np.max(np.array([x.shape for x in aucFold]), axis=0)]))
  aucFoldPad[:,:,:,:]=np.nan
  for aucFoldInd in range(0, len(aucFold)):
    aucFoldPad[aucFoldInd,:,0:aucFold[aucFoldInd].shape[1],:]=aucFold[aucFoldInd]
  aucFold=aucFoldPad
  aucMean=np.nanmean(aucFold, axis=0)
  rm=np.nan_to_num(np.array([pd.stats.moments.rolling_mean(aucMean[ind], center=True, window=100) for ind in range(0, aucMean.shape[0])]))
  paramInd=rm.max(1).mean(1).argmax()
  bestIterPerTask=rm[paramInd].argmax(0)
  
  return (paramInd, bestIterPerTask, rm, aucFold)

def bestSettingsSimple(perfFiles, nrParams, takeMinibatch=[-1,-1,-1]):
  aucFold=[]
  for foldInd in range(0, len(perfFiles)):
    innerFold=-1
    aucParam=[]
    for paramNr in range(0, nrParams):
      #try:
      saveFile=open(perfFiles[foldInd][paramNr], "rb")
      aucRun=pickle.load(saveFile)
      saveFile.close()
      #except:
      #  pass
      if(len(aucRun)>0):
        if takeMinibatch[foldInd]<len(aucRun):
          aucParam.append(aucRun[takeMinibatch[foldInd]])
        else:
          aucParam.append(aucRun[-1])
    
    aucParam=np.array(aucParam)
    
    if(len(aucParam)>0):
      aucFold.append(aucParam)
  aucFold=np.array(aucFold)
  aucMean=np.nanmean(aucFold, axis=0)
  paramInd=np.nanmean(aucMean, axis=1).argmax()
  
  return (paramInd, aucMean, aucFold)

def getOuterPerformance(perfFiles, takeMinibatch):
  aucParam=[]
  aucParamExt=[]
  for paramNr in range(0, len(perfFiles)):
    saveFile=open(perfFiles[paramNr], "rb")
    aucRun=pickle.load(saveFile)
    saveFile.close()
    if(len(aucRun)>0):
      if takeMinibatch<len(aucRun):
        aucParam.append(aucRun[takeMinibatch])
        aucParamExt.append(np.array(aucRun)[0:(takeMinibatch+1)])
      else:
        aucParam.append(aucRun[-1])
        aucRun.extend([aucRun[-1]]*(takeMinibatch-len(aucRun)))
        aucParamExt.append(np.array(aucRun)[0:(takeMinibatch+1)])

  aucParam=np.array(aucParam)

  maxShape0=max([x.shape[0] for x in aucParamExt])
  shape1=aucParamExt[0].shape[1]
  aucParamExtCopy=np.zeros(shape=(len(aucParamExt), maxShape0, shape1))
  aucParamExtCopy[:,:,:]=np.nan
  for paramNr in range(0, len(perfFiles)):
    aucParamExtCopy[paramNr,0:aucParamExt[paramNr].shape[0],:]=aucParamExt[paramNr]
  aucParamExt=aucParamExtCopy
  
  return(aucParam, aucParamExt)

def getInnerPerformance(perfFiles, takeMinibatch):
  aucFold=[]
  aucFoldExt=[]
  for foldInd in range(0, len(perfFiles)):
    aucParam=[]
    aucParamExt=[]
    for paramNr in range(0, len(perfFiles[0])):
      saveFile=open(perfFiles[foldInd][paramNr], "rb")
      aucRun=pickle.load(saveFile)
      saveFile.close()
      if(len(aucRun)>0):
        if takeMinibatch[foldInd]<len(aucRun):
          aucParam.append(aucRun[takeMinibatch[foldInd]])
          aucParamExt.append(np.array(aucRun)[0:(takeMinibatch[foldInd]+1)])
        else:
          aucParam.append(aucRun[-1])
          aucRun.extend([aucRun[-1]]*(takeMinibatch[foldInd]-len(aucRun)))
          aucParamExt.append(np.array(aucRun)[0:(takeMinibatch[foldInd]+1)])
    
    aucParam=np.array(aucParam)
    if(len(aucParam)>0):
      aucFold.append(aucParam)
    
    maxShape0=max([x.shape[0] for x in aucParamExt])
    shape1=aucParamExt[0].shape[1]
    aucParamExtCopy=np.zeros(shape=(len(aucParamExt), maxShape0, shape1))
    aucParamExtCopy[:,:,:]=np.nan
    for paramNr in range(0, len(perfFiles[0])):
      aucParamExtCopy[paramNr,0:aucParamExt[paramNr].shape[0],:]=aucParamExt[paramNr]
    aucParamExt=aucParamExtCopy
    if(len(aucParamExt)>0):
      aucFoldExt.append(aucParamExt)

  aucFold=np.array(aucFold)

  shape0=aucFoldExt[0].shape[0]
  maxShape1=max([x.shape[1] for x in aucFoldExt])
  shape2=aucFoldExt[0].shape[2]
  aucFoldExtCopy=np.zeros(shape=(len(aucFoldExt), shape0, maxShape1, shape2))
  aucFoldExtCopy[:,:,:,:]=np.nan
  for foldInd in range(0, len(aucFoldExt)):
    aucFoldExtCopy[foldInd,:,0:aucFoldExt[foldInd].shape[1],:]=aucFoldExt[foldInd]
  aucFoldExt=aucFoldExtCopy

  aucMean=np.nanmean(aucFold, axis=0)
  aucMeanExt=np.nanmean(aucFoldExt, axis=0)
  
  return(aucMean, aucMeanExt)



def getInnerEndPerformance(perfFiles):
  aucFold=[]
  for foldInd in range(0, len(perfFiles)):
    aucParam=[]
    for paramNr in range(0, len(perfFiles[0])):
      saveFile=open(perfFiles[foldInd][paramNr], "rb")
      aucRun=pickle.load(saveFile)
      saveFile.close()
      aucParam.append(aucRun)
    aucParam=np.array(aucParam)
    if(len(aucParam)>0):
      aucFold.append(aucParam)
  aucFold=np.array(aucFold)
  aucMean=np.nanmean(aucFold, axis=0)
  
  return(aucMean)