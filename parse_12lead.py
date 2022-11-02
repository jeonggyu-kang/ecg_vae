
import argparse
import functools
import glob
import matplotlib.pyplot as plt
import mmap
import numpy as np
import os
import pandas as pd
import pickle 
import re
import tqdm
import gzip
import gc
import dask.dataframe as dd
import sklearn.model_selection

from dask.distributed import Client, LocalCluster
from multiprocessing import Pool

diagnosis_pattern = r'(?<=소견 코드\[\d\] : )\d{4}'
diagnosis_matcher = re.compile(diagnosis_pattern)

lead1_pattern = r'(?<=Lead 0 :)( -?\d+)* \n'
lead1_matcher = re.compile(lead1_pattern)

lead2_pattern = r'(?<=Lead 1 :)( -?\d+)* \n'
lead2_matcher = re.compile(lead2_pattern)

leadavf_pattern = r'(?<=Lead aVF :)( -?\d+)* \n'
leadavf_matcher = re.compile(leadavf_pattern)

leadavl_pattern = r'(?<=Lead aVL :)( -?\d+)* \n'
leadavl_matcher = re.compile(leadavl_pattern)

patient_pattern = r'(?<=PATID : )\d*'
patient_matcher = re.compile(patient_pattern)

date_pattern = r'(?<=DATE : )\d*-\d*-\d*'
date_matcher = re.compile(date_pattern)

sample_pattern = r'-?\d+'
sample_matcher = re.compile(sample_pattern)

categories = pd.read_csv("diagnostic_counts_final_category.csv")
categories = categories[["diagnosis", "arrhythmia_code"]].drop_duplicates()
categories.set_index("diagnosis", inplace=True)

def fukuda_classification(text):
    if code in [4124, 4136, 4146, 4156, 8044, 8314, 8414, 8424, 8434, 8446, 8454,
                8466, 8476, 8486, 8516, 8526, 8616, 8626, 8636, 8646, 8656, 8664,
                8676, 8686, 8716, 8726]:
        return 1
    else:
        return 0
    return

def diagnostic_classification(diagnostics):
    """
        Classification by Dr. Kang. Refer to 'category.xlsx'.
    
        code 0: normal
        code 1: arrythmia
        code -1: exclude
    """
    diagnoses = np.array(categories.loc[diagnostics, 'arrhythmia_code'])
    if any(diagnoses == 9.0):
        return -1
    elif any(~np.isnan(diagnoses)):
        return 1
    else:
        return 0

def parse_diagnosis(string):
    diagnosis_codes = diagnosis_matcher.findall(string)
    return str(diagnosis_codes[0])

def parse_date(string):
    dates = date_matcher.findall(string)
    return str(dates[0])

def parse_patient(string):
    patient_codes = patient_matcher.findall(string)
    return int(patient_codes[0])

def parse_signal(matcher, string):
    signal  = matcher.search(string).group()
    samples = sample_matcher.findall(signal)
    samples = map(int, samples)
    return np.array(list(samples))

def parse_signals(string):
    signals = {}
    signals["lead1"]   = parse_signal(lead1_matcher,   string)
    signals["lead2"]   = parse_signal(lead2_matcher,   string)
    signals["leadavf"] = parse_signal(leadavf_matcher, string)
    signals["leadavl"] = parse_signal(leadavl_matcher, string)
    return signals

def parse_file(fpath):
    io     = open(fpath, 'r', encoding='EUC-KR') 
    string = io.read()
    io.close()
    try:
        diagnosis = parse_diagnosis(string)
        date      = parse_date(string)
        patient   = parse_patient(string)
        data      = {"fukuda"  : diagnosis,
                     "date"    : date,
                     "patient" : patient}
        data.update(parse_signals(string))
        return data
    except:
        return {}

def make_database(path, pool, name):
    if os.path.isfile("{}.gzip".format(name)):
        return

    filelist = glob.glob(path)
    filelist = list(filter(lambda name : name[-4:] == ".log", filelist))
    n_files  = len(filelist)
    # df       = pd.DataFrame(columns=['patient',
    #                                  'date',
    #                                  'lead1',
    #                                  'lead2',
    #                                  'leadavf',
    #                                  'leadavl',
    #                                  'fukuda'])

    pbar = tqdm.tqdm(pool.imap_unordered(parse_file, filelist), total=len(filelist))
    data = {}
    for idx, parsed in enumerate(pbar):
        if len(parsed) > 0:
            data[idx] = parsed
        pbar.set_description("number of dataframe rows = {}".format(len(data)))
    df = pd.DataFrame.from_dict(data, "index")
    df.to_pickle("{}.gzip".format(name))
    return

def add_cardio_diagnosis(file, pool, time_start, time_end):
    ecg_log   = pd.read_pickle(file)
    ecg_log.set_index(['patient', 'date'], inplace=True, drop=False)
    ecg_log.sort_index(ascending=True)

    diagnoses = pd.read_csv('clinical_diagnostics.csv',
                            delimiter=',',
                            encoding='EUC-KR',
                            encoding_errors='ignore',
                            index_col=False)
    diagnoses['등록번호'] = [int(patid_corrupt[1:]) for patid_corrupt in diagnoses['등록번호']]
    diagnoses.rename(columns={"등록번호": "patient", "검진일자":"date"},
                     errors="raise",
                     inplace=True)
    diagnoses = diagnoses[(diagnoses["date"] >= time_start) & (diagnoses["date"] <= time_end)]

    def add_cardio_work(diagnose):
        try:
            entries = ecg_log.loc[(diagnose["patient"], diagnose["date"]), :]
            new_entry = dict(entries.iloc[-1,:])
            diagnose  = diagnose[["심전도", "UA20041", "UA20042", "UA20043"]]
            diagnose  = list(diagnose[~diagnose.isnull()])
            new_entry.update({"cadio": diagnose})
            return new_entry
        except:
            return {}

    pbar = tqdm.tqdm(diagnoses.iterrows(), total=len(diagnoses))
    data = {}
    idx  = 0
    for _, diagnose in pbar:
        new_entry = add_cardio_work(diagnose)
        if len(new_entry) > 0:
            data[idx] = new_entry
            idx += 1
        pbar.set_description("number of dataframe rows = {}".format(len(data)))

    df = pd.DataFrame.from_dict(data, "index")
    df.to_pickle(file)
    return

def exclude_entry(ecg_log):
    arrhythm_flag = np.array(list(
        map(diagnostic_classification, list(ecg_log['cadio']))))
    return arrhythm_flag < 0

def find_arrhythm_entries(ecg_log):
    arrhythm_flag = np.array(list(
        map(diagnostic_classification, list(ecg_log['cadio']))))
    return arrhythm_flag > 0

def classify_patients(ecg_log):
    arrhythm_flag = np.array(list(
        map(diagnostic_classification, list(ecg_log['cadio']))))
    arrhythm_patients = ecg_log[arrhythm_flag > 0]['patient']
    arrhythm_patients = np.unique(arrhythm_patients)
    normal_patients   = ecg_log[arrhythm_flag == 0]['patient']
    normal_patients   = np.unique(normal_patients)
    return normal_patients, arrhythm_patients

# def select_test_patients():
#     ecg_log = pd.read_pickle( "log_ecg_2018.gzip")
#     test_normal_patients, test_arrhythm_patients = classify_patients(ecg_log)
#     test_normal_patients = np.random.choice(test_normal_patients, size=1000)
#     ecg_log = []
#     gc.collect()
#     return test_normal_patients, test_arrhythm_patients

def classify_all_patients():
    arrhythm_patients = []
    normal_patients   = []

    for name in tqdm.tqdm(["../SSD/log_ecg_2014.gzip",
                           "../SSD/log_ecg_2015.gzip",
                           "../SSD/log_ecg_2016.gzip",
                           "../SSD/log_ecg_2017.gzip"],
                          total=4):
        ecg_log = pd.read_pickle(name)
        ecg_log = ecg_log[~exclude_entry(ecg_log)]
        normal_patients_subset, arrhythm_patients_subset = classify_patients(ecg_log)
        arrhythm_patients.extend(arrhythm_patients_subset)
        normal_patients.extend(normal_patients_subset)
        ecg_log = []
        gc.collect()

    normal_patients   = np.unique(normal_patients)
    arrhythm_patients = np.unique(arrhythm_patients)
    normal_patients   = np.setdiff1d(normal_patients, arrhythm_patients)
    return normal_patients, arrhythm_patients

def gather_signals(patient_list):
    all_signals = []
    pbar    = tqdm.tqdm(["../SSD/log_ecg_2014.gzip",
                         "../SSD/log_ecg_2015.gzip",
                         "../SSD/log_ecg_2016.gzip",
                         "../SSD/log_ecg_2017.gzip"],
                        total=4)
    for name in pbar:
        ecg_log = pd.read_pickle(name)
        ecg_log = ecg_log[~exclude_entry(ecg_log)]
        ecg_log.set_index('patient', inplace=True, drop=False)
        ecg_log.sort_index(ascending=True)

        #is_arrhythm = find_arrhythm_entries(ecg_log)
        is_target   = ecg_log['patient'].isin(patient_list)

        ecg_log = ecg_log[is_target]
        signals = [signal.astype(np.int16) for signal in list(ecg_log["lead1"])]
        all_signals.extend(signals)
        pbar.set_description("number of signals = {}".format(len(all_signals)))
        signals = []
        ecg_log = []
        gc.collect()
    return all_signals

def filter_entries(patient_list):
    filtered_data = {}
    pbar    = tqdm.tqdm(["../SSD/log_ecg_2014.gzip",
                         "../SSD/log_ecg_2015.gzip",
                         "../SSD/log_ecg_2016.gzip",
                         "../SSD/log_ecg_2017.gzip"],
                        total=4)
    for name in pbar:
        ecg_log = pd.read_pickle(name)
        ecg_log = ecg_log[~exclude_entry(ecg_log)]
        subset  = ecg_log[ecg_log['patient'].isin(patient_list)].to_dict('index')
        filtered_data.update(subset)
        pbar.set_description("number of valid entries = {}".format(len(filtered_data)))

        ecg_log = []
        gc.collect()

    df = pd.DataFrame.from_dict(filtered_data, 'index')
    return df

def train_dev_test_split(patients):
    train, valid_test = sklearn.model_selection.train_test_split(
        patients,
        train_size=0.8,
        test_size=0.2)
    valid, test = sklearn.model_selection.train_test_split(
        valid_test,
        train_size=0.5,
        test_size=0.5)
    return train, valid, test

def make_dataset():
    mode = "all_lead1"

    test_set = pd.read_pickle("all_lead1_gold_test.pickle")
    test_patients = np.unique(test_set["patient"])

    normal_patients, arrhythm_patients = classify_all_patients()
    gc.collect()

    normal_train, normal_valid, normal_test       = train_dev_test_split(normal_patients)
    arrhythm_train, arrhythm_valid, arrhythm_test = train_dev_test_split(arrhythm_patients)

    valid_patients = np.union1d(normal_valid, arrhythm_valid)
    train_patients = np.union1d(normal_train, arrhythm_train)
    test_patients  = np.union1d(normal_test,  arrhythm_test)

    print("# of arrhythmia patients in train set = {}, valid set = {}, test set = {}".format(
        len(arrhythm_train),
        len(arrhythm_valid),
        len(arrhythm_test)))
    print("# of normal patients in train set = {}, valid set = {}, test set = {}".format(
        len(normal_train),
        len(normal_valid),
        len(normal_test)))

    print("-- gathering train data signals")
    train_signals = gather_signals(train_patients)

    with open('all_lead1_train.pickle', 'wb') as f:
        pickle.dump(train_signals, f)

    train_signals  = []
    gc.collect()

    print("-- gathering validation data signals")
    df = filter_entries(valid_patients)
    df.to_pickle("all_lead1_valid.gzip")

    print("-- gathering test data signals")
    df = filter_entries(test_patients)
    df.to_pickle("all_lead1_test.gzip")

        # valid_arrhythm_patients

        # valid_arrhythm_signals = gather_signals(valid_patients, True)
        # valid_normal_signals   = gather_signals(valid_patients, False)

        # with open('all_lead1_valid_normal.pickle', 'wb') as f:
        #     pickle.dump(valid_normal_signals, f)
        # with open('all_lead1_valid_arrhythm.pickle', 'wb') as f:
        #     pickle.dump(valid_arrhythm_signals, f)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_fukuda',     action='store_true', help='Parse Fukuda log files')
    parser.add_argument('--add_diagnosis',    action='store_true', help='Add cardiologist diagnostics')
    parser.add_argument('--make_dataset',     action='store_true', help='Make dataset pickle files')
    parser.add_argument('--merge_dataframes', action='store_true', help='Merge dataframes into a single parquet database')
    opt  = parser.parse_args()

    pool = Pool(28)
    if opt.parse_fukuda:
        make_database('/home/arrhythmia/data/log_ecg_2018/*', pool, "log_ecg_2018")
        #make_database('/home/arrhythmia/data/log_ecg_2017/*', pool, "log_ecg_2017")
        #make_database('/home/arrhythmia/data/log_ecg_2016/*', pool, "log_ecg_2016")
        #make_database('/home/arrhythmia/data/log_ecg_2015/*', pool, "log_ecg_2015")
        #make_database('/home/arrhythmia/data/log_ecg_2014/*', pool, "log_ecg_2014")

    if opt.add_diagnosis:
        add_cardio_diagnosis('log_ecg_2018.gzip', pool, "2018-01-01", "2018-12-31")
        # add_cardio_diagnosis('log_ecg_2017.gzip', pool, "2017-01-01", "2017-12-31")
        # add_cardio_diagnosis('log_ecg_2016.gzip', pool, "2016-01-01", "2016-12-31")
        # add_cardio_diagnosis('log_ecg_2015.gzip', pool, "2015-01-01", "2015-12-31")
        # add_cardio_diagnosis('log_ecg_2014.gzip', pool, "2014-01-01", "2014-12-31")

    if opt.merge_dataframes:
        merge_dataframes()

    if opt.make_dataset:
        make_dataset()
    return

if __name__ == "__main__":
    main()
