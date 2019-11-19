#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

if continueComputations:
  if computeTestPredictions:
    if compPerformanceTest:
      saveFilename=savePrefix+".test.aucMB.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTestAUC=pickle.load(saveFile)
        saveFile.close()
      
      saveFilename=savePrefix+".test.apMB.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTestAP=pickle.load(saveFile)
        saveFile.close()
      
      saveFilename=savePrefix+".test.aucEnd.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        sumTestAUC=pickle.load(saveFile)
        saveFile.close()
      
      saveFilename=savePrefix+".test.apEnd.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        sumTestAP=pickle.load(saveFile)
        saveFile.close()
  
  if computeTrainPredictions:
    if compPerformanceTrain:
      saveFilename=savePrefix+".train.aucMB.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTrainAUC=pickle.load(saveFile)
        saveFile.close()

      saveFilename=savePrefix+".train.apMB.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTrainAP=pickle.load(saveFile)
        saveFile.close()
        
      saveFilename=savePrefix+".train.aucEnd.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        sumTrainAUC=pickle.load(saveFile)
        saveFile.close()

      saveFilename=savePrefix+".train.apEnd.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        sumTrainAP=pickle.load(saveFile)
        saveFile.close()

if logPerformanceAtBestIter:
  saveFilename=savePrefix+".test.aucBI.npy"
  if os.path.isfile(saveFilename):
    reportAUCBestIter=np.load(saveFilename)
  
  saveFilename=savePrefix+".test.apBI.npy"
  if os.path.isfile(saveFilename):
    reportAPBestIter=np.load(saveFilename)  

if savePredictionsAtBestIter:
  if useDenseOutputNetPred:
    saveFilename=savePrefix+".test.predBI.pckl"
    if os.path.isfile(saveFilename):
      saveFile=open(saveFilename, "rb")
      predDenseBestIter=pickle.load(saveFile)
      saveFile.close()
  else:
    saveFilename=savePrefix+".test.predBI.pckl"
    if os.path.isfile(saveFilename):
      saveFile=open(saveFilename, "rb")
      predSparseBestIter=pickle.load(saveFile)
      saveFile.close()

saveFilename=savePrefix+".trainInfo.pckl"
if os.path.isfile(saveFilename):
  saveFile=open(saveFilename, "rb")
  startEpoch=pickle.load(saveFile)
  minibatchCounterTrain=pickle.load(saveFile)
  minibatchCounterTest=pickle.load(saveFile)
  minibatchReportNr=pickle.load(saveFile)
  saveFile.close()

if "session" in dir():
  saveFilename=savePrefix+".trainModel.meta"
  if os.path.isfile(saveFilename):
    saveFilename=savePrefix+".trainModel"
    tf.train.Saver().restore(session, saveFilename)
