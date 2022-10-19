
import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc

from augmentations import ecg_augmentation, ecg_augmentation_partial

def diagnostic_classification(diagnostics):
    """
        Classification by Dr. Kang. Refer to 'category.xlsx'.
    
        code 0: normal
        code 1: arrythmia
    """
    if any(map(lambda diagnostic : diagnostic in [
            '심실조기수축',
            '빈맥',
            '심장접합부박동',
            '다발성심실조기수축',
            '심방조기수축',
            '심방세동',
            '빈번한심방조기수축',
            '심방조기수축',
            '심장접합부박동',
            '인공심장박동기부착',
            '비지속성심실빈맥',
            '동정지의심',
            '심방세동',
            '심장접합부박동의증',
            '심장접합부박동',
            '심실조기수축',
            '심장접합부이탈박동',
            '이소성심박동의증',
            '동방블럭',
            '발작성빈맥의증',
            '모비츠타입2방실블럭',
            '부정맥',
            '심장정밀검사요함',
            '심장접합부이탈박동의증',
            '심장접합부이탈박동',
            '심실성빈맥',
            '심실상성 빈맥의증',
            '심방조기수축의증',
            '다발성심실조기수축',
            '빈번한심방조기수축',
            '동방블럭의심',
            '동정지',
            '비지속성심실빈맥',
            '심방빈맥',
            '심실성빈맥',
            '심장접합부조기박동',
            '동정지',
            '동기능부전증후군의심',
            '다소성심방빈맥',
            '판정참고',],
               diagnostics)):
        return 1
    else:
        return 0

def read_file(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def find_arrhythm_entries(ecg_log):
    arrhythm_flag = np.array(list(
        map(diagnostic_classification, list(ecg_log['cadio']))))
    return arrhythm_flag > 0

class FukudaECGDataset():
    def __init__(self, mode="train", augmentation=None):
        self.mode = mode

        if mode == "train":
            data              = read_file('../../all_lead1_train.pickle');
            self.signals      = [d.astype(np.float32) for d in data]
            self.augmentation = augmentation
            #signals      = [np.random.randn(5000).astype(np.float32) for i in range(10000)]
            #self.signals = signals

        elif mode == "valid_normal":
            data      = pd.read_pickle('../../all_lead1_valid.gzip')
            data      = data[~find_arrhythm_entries(data)]
            signals   = [signal.astype(np.float32) for signal in data["lead1"]]
            diagnoses = list(data["cadio"])
            bundles   = zip(signals, diagnoses)
            bundles   = list(filter(lambda bundle: len(bundle[0]) == 5000, bundles))
            signals, diagnoses = zip(*bundles)
            self.signals   = signals
            self.diagnoses = diagnoses
            data = []
            gc.collect()

        elif mode == "valid_arrhythm":
            data      = pd.read_pickle('../../all_lead1_valid.gzip')
            data      = data[find_arrhythm_entries(data)]
            signals   = [signal.astype(np.float32) for signal in data["lead1"]]
            diagnoses = list(data["cadio"])
            bundles   = zip(signals, diagnoses)
            bundles   = list(filter(lambda bundle: len(bundle[0]) == 5000, bundles))
            signals, diagnoses = zip(*bundles)
            self.signals   = signals
            self.diagnoses = diagnoses
            data = []
            gc.collect()

        self.augmentation = augmentation

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx][0:5000]
        mu     = np.mean(signal)
        sigma  = np.std(signal)
        signal = (signal - mu) / (sigma + 1e-10)

        if self.augmentation == "full":
            signal = ecg_augmentation(signal)
            mu     = np.mean(signal)
            sigma  = np.std(signal)
            signal = (signal - mu) / (sigma + 1e-10)
        elif self.augmentation == "partial":
            signal = ecg_augmentation_partial(signal)
            mu     = np.mean(signal)
            sigma  = np.std(signal)
            signal = (signal - mu) / (sigma + 1e-10)

        signal   = np.expand_dims(signal, 0)
        diagnose = ""

        if ((self.mode == "valid_normal")
            or (self.mode == "valid_arrhythm")):
            diagnose = str(self.diagnoses[idx])
            
        return signal, diagnose
        
