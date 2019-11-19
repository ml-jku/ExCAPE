#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

import numpy as np
import scipy
import pickle
import pandas as pd



saveFilename="auc1.pckl"
saveFile=open(saveFilename, "rb")
testAUC1=pickle.load(saveFile)
saveFile.close()

saveFilename="auc2.pckl"
saveFile=open(saveFilename, "rb")
testAUC2=pickle.load(saveFile)
saveFile.close()

saveFilename="auc3.pckl"
saveFile=open(saveFilename, "rb")
testAUC3=pickle.load(saveFile)
saveFile.close()

print(np.round(np.mean((testAUC1+testAUC2+testAUC3)/3),2))
print(np.round(np.std((testAUC1+testAUC2+testAUC3)/3),2))
myAUC=pd.DataFrame({"dnn_of1_auc": testAUC1, "dnn_of2_auc": testAUC2, "dnn_of3_auc": testAUC3})
myAUC.to_csv("auc.csv")


saveFilename="f1.pckl"
saveFile=open(saveFilename, "rb")
testF11=pickle.load(saveFile)
saveFile.close()

saveFilename="f2.pckl"
saveFile=open(saveFilename, "rb")
testF12=pickle.load(saveFile)
saveFile.close()

saveFilename="f3.pckl"
saveFile=open(saveFilename, "rb")
testF13=pickle.load(saveFile)
saveFile.close()

print(np.round(np.mean((testF11+testF12+testF13)/3),2))
print(np.round(np.std((testF11+testF12+testF13)/3),2))
myF1=pd.DataFrame({"dnn_of1_f1": testF11, "dnn_of2_f1": testF12, "dnn_of3_f1": testF13})
myF1.to_csv("f1.csv")



saveFilename="kappa1.pckl"
saveFile=open(saveFilename, "rb")
testKappa1=pickle.load(saveFile)
saveFile.close()

saveFilename="kappa2.pckl"
saveFile=open(saveFilename, "rb")
testKappa2=pickle.load(saveFile)
saveFile.close()

saveFilename="kappa3.pckl"
saveFile=open(saveFilename, "rb")
testKappa3=pickle.load(saveFile)
saveFile.close()

print(np.round(np.mean((testKappa1+testKappa2+testKappa3)/3),2))
print(np.round(np.std((testKappa1+testKappa2+testKappa3)/3),2))
myKappa=pd.DataFrame({"dnn_of1_kappa": testKappa1, "dnn_of2_kappa": testKappa2, "dnn_of3_kappa": testKappa3})
myKappa.to_csv("kappa.csv")






