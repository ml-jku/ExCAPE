#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

#results from step2B in order to obtain:
testDenseOutput
predDenseTest

#To this for each of the outer folds using "ofolds" from 0 to 2 one after each other and save under auc1, f1, kappa,   auc2, f2,... (copy code from below)


testKappa=[]
testF1=[]
testAUC=[]
for selTarget in range(testDenseOutput.shape[1]):
  myidx=np.abs(testDenseOutput[:,selTarget])>0.5
  selPred=predDenseTest[myidx, selTarget]
  selTest=testDenseOutput[myidx, selTarget]
  testKappa.append(sklearn.metrics.cohen_kappa_score(testDenseOutput[myidx, selTarget], np.where(predDenseTest[myidx, selTarget]>=0.5, 1, -1)))
  testF1.append(sklearn.metrics.f1_score(testDenseOutput[myidx, selTarget], np.where(predDenseTest[myidx, selTarget]>=0.5, 1, -1)))
  testAUC.append(sklearn.metrics.roc_auc_score(testDenseOutput[myidx, selTarget], predDenseTest[myidx, selTarget]))

#and save in pickle files

import pickle

saveFilename="auc1.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testAUC), saveFile)
saveFile.close()
saveFilename="f1.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testF1), saveFile)
saveFile.close()
saveFilename="kappa1.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testKappa), saveFile)
saveFile.close()

saveFilename="auc2.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testAUC), saveFile)
saveFile.close()
saveFilename="f2.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testF1), saveFile)
saveFile.close()
saveFilename="kappa2.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testKappa), saveFile)
saveFile.close()

saveFilename="auc3.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testAUC), saveFile)
saveFile.close()
saveFilename="f3.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testF1), saveFile)
saveFile.close()
saveFilename="kappa3.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(np.array(testKappa), saveFile)
saveFile.close()





