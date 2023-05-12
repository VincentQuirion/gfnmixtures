import pickle
from typing import Any

import numpy as np
import torch

from rdkit.Chem import AllChem
from rdkit import DataStructs, Chem

class KMeansClassifier:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.km = pickle.load(f)

    def kmeans_classify(self, inputs):
        km = self.km

        fp_list = []
        for mol in inputs:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1,), int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_list.append(arr)
        fps = torch.tensor(np.array(fp_list))
        return km.predict(fps)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        classes = self.kmeans_classify(*args, **kwds)
        return classes