#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public



if not os.path.exists(savePrefix):
  os.mkdir(savePrefix)
tmpSavePrefix=savePrefix+"/"+savePrefix0



if computeTestPredictions:
  if compPerformanceTest:
    saveFilename=tmpSavePrefix+".test.aucMB.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTestAUC, saveFile)
    saveFile.close()

    saveFilename=tmpSavePrefix+".test.apMB.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTestAP, saveFile)
    saveFile.close()
    
    saveFilename=tmpSavePrefix+".test.aucEnd.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(sumTestAUC, saveFile)
    saveFile.close()

    saveFilename=tmpSavePrefix+".test.apEnd.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(sumTestAP, saveFile)
    saveFile.close()

if computeTrainPredictions:
  if compPerformanceTrain:
    saveFilename=tmpSavePrefix+".train.aucMB.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTrainAUC, saveFile)
    saveFile.close()

    saveFilename=tmpSavePrefix+".train.apMB.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTrainAP, saveFile)
    saveFile.close()
    
    saveFilename=tmpSavePrefix+".train.aucEnd.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(sumTrainAUC, saveFile)
    saveFile.close()

    saveFilename=tmpSavePrefix+".train.apEnd.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(sumTrainAP, saveFile)
    saveFile.close()

saveFilename=tmpSavePrefix+".trainInfo.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(epoch, saveFile)
pickle.dump(minibatchCounterTrain, saveFile)
pickle.dump(minibatchCounterTest, saveFile)
pickle.dump(minibatchReportNr, saveFile)
saveFile.close()

saveFilename=tmpSavePrefix+".trainModel"
tf.train.Saver().save(session, saveFilename)

saveFilename=tmpSavePrefix+"."+finMark+".pckl"
saveFile=open(saveFilename, "wb")
finNr=0
pickle.dump(finNr, saveFile)
saveFile.close()



os.system("rm "+savePrefix+"/checkpoint")
for file in os.listdir(savePrefix):
  if os.path.exists(savePath+file):
    os.remove(savePath+file)
import glob
import shutil
[shutil.move(myfile, savePath) for myfile in glob.glob(savePrefix+"/*")]
#os.system("mv "+savePrefix+"/* "+savePath)
os.rmdir(savePrefix)