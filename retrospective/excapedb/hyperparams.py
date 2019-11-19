#Copyright (C) 2016-2019 Andreas Mayr, Guenter Klambauer, Thomas Unterthiner, Sepp Hochreiter
#Licensed under original BSD License (see LICENSE-ExCAPE at base directory) for members of the Horizon 2020 Project ExCAPE (Grant Agreement no. 671555)
#Licensed under GNU General Public License v3.0 (see LICENSE at base directory) for the general public

import itertools
import numpy as np
import pandas as pd

dictionary0 = {
  'basicArchitecture': [('SELUScalingDropout', 'seluScaling'), ('ReLUScalingDropout', 'reluScaling')],
  'l2Penalty': [0.0],
  'learningRate': [0.01, 0.1],
  'l1Penalty': [0.0],
  'layerForm': ["rect"],
  'idropout': [0.0, 0.2],
  'dropout': [0.5],
  'nrStart': [1024, 2048, 4096],
  'nrLayers': [3],
  'mom': [0.0]
}

dictionary1 = {
  'basicArchitecture': [('SELUScalingDropout', 'seluScaling'), ('ReLUScalingDropout', 'reluScaling')],
  'l2Penalty': [0.0],
  'learningRate': [0.01, 0.1],
  'l1Penalty': [0.0],
  'layerForm': ["rect"],
  'idropout': [0.0, 0.2],
  'dropout': [0.5],
  'nrStart': [2048],
  'nrLayers': [2, 4],
  'mom': [0.0]
}

hyperParams0 = pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams1 = pd.DataFrame(list(itertools.product(*dictionary1.values())), columns=dictionary1.keys())
hyperParams=pd.concat([hyperParams0, hyperParams1], axis=0)
hyperParams.index=np.arange(len(hyperParams.index.values))