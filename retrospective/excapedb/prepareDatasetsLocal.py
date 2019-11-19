#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

if not (denseOutputData is None):
  trainDenseOutput=denseOutputData[sampleAnnInd[trainSamples].values].copy()
  testDenseOutput=denseOutputData[sampleAnnInd[testSamples].values].copy()
  trainPos=(trainDenseOutput > 0.5).sum(axis=0)
  trainNeg=(trainDenseOutput < -0.5).sum(axis=0)

if not (sparseOutputData is None):
  trainSparseOutput=sparseOutputData[sampleAnnInd[trainSamples].values].copy()
  trainSparseOutputTransposed=trainSparseOutput.copy().T.tocsr()
  trainSparseOutputTransposed.sort_indices()
  testSparseOutput=sparseOutputData[sampleAnnInd[testSamples].values].copy()
  testSparseOutputTransposed=testSparseOutput.copy().T.tocsr()
  testSparseOutputTransposed.sort_indices()
  trainPos=np.array([np.sum(trainSparseOutputTransposed[x].data > 0.5) for x in range(trainSparseOutputTransposed.shape[0])])
  trainNeg=np.array([np.sum(trainSparseOutputTransposed[x].data < -0.5) for x in range(trainSparseOutputTransposed.shape[0])])

trainProp=trainPos/(trainPos+trainNeg)
trainBias=np.log(trainProp/(1.0-trainProp))
trainBias[np.logical_not(np.logical_and(trainPos>10, trainNeg>10))]=0.0
#trainBias[:]=0.0 #comment/uncomment if bias at output should reflect imbalance level



if savePredictionsAtBestIter:
  if computeTestPredictions:
    if useDenseOutputNetPred:
      predDenseBestIter=-np.ones((len(testSamples), nrOutputTargets))
    else:
      predSparseBestIter=testSparseOutput.copy().astype(np.float32)
      predSparseBestIter.data[:]=-1

if logPerformanceAtBestIter:
  if computeTestPredictions:
    reportAUCBestIter=np.zeros(nrOutputTargets)
    reportAPBestIter=np.zeros(nrOutputTargets)



nrDenseFeatures=0
if not (denseInputData is None):
  trainDenseInput=denseInputData[denseSampleIndex[trainSamples].values].copy()
  testDenseInput=denseInputData[denseSampleIndex[testSamples].values].copy()
  nrDenseFeatures=trainDenseInput.shape[1]

  if normalizeLocalDense:
    trainDenseMean1=np.nanmean(trainDenseInput, 0)
    trainDenseStd1=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean1)/trainDenseStd1
    trainDenseInput=np.tanh(trainDenseInput)
    trainDenseMean2=np.nanmean(trainDenseInput, 0)
    trainDenseStd2=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean2)/trainDenseStd2
    
    testDenseInput=(testDenseInput-trainDenseMean1)/trainDenseStd1
    testDenseInput=np.tanh(testDenseInput)
    testDenseInput=(testDenseInput-trainDenseMean2)/trainDenseStd2
  
  trainDenseInput=np.nan_to_num(trainDenseInput)
  testDenseInput=np.nan_to_num(testDenseInput)



nrSparseFeatures=0
if not (sparseInputData is None):
  trainSparseInput=sparseInputData[sparseSampleIndex[trainSamples].values].copy()
  featSel=trainSparseInput.getnnz(axis=0)/float(trainSparseInput.shape[0])> sparsenesThr
  mydenseInputData=sparseInputData[:, featSel].tocsr()
  
  trainDenseInput=mydenseInputData[sparseSampleIndex[trainSamples].values].A
  testDenseInput=mydenseInputData[sparseSampleIndex[testSamples].values].A
  nrDenseFeatures=trainDenseInput.shape[1]
  
  if normalizeLocalDense:
    trainDenseMean1=np.nanmean(trainDenseInput, 0)
    trainDenseStd1=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean1)/trainDenseStd1
    trainDenseInput=np.tanh(trainDenseInput)
    trainDenseMean2=np.nanmean(trainDenseInput, 0)
    trainDenseStd2=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean2)/trainDenseStd2
    
    testDenseInput=(testDenseInput-trainDenseMean1)/trainDenseStd1
    testDenseInput=np.tanh(testDenseInput)
    testDenseInput=(testDenseInput-trainDenseMean2)/trainDenseStd2
  
  trainDenseInput=np.nan_to_num(trainDenseInput)
  testDenseInput=np.nan_to_num(testDenseInput)
