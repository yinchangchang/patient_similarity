#!/usr/bin/env python
import numpy as np
from stdFunc import *


# word_embedding = np.load('../data/word2vec_50.npy')
# print word_embedding

def gen_fid_index_dict():
    fid_index_dict = { }
    model_npy = np.zeros([8628, 50 ])
    for i,line in enumerate(open('../data/model_50.vector')):
        data = line.strip().split()
        if len(line) > 3:
            fid = data[0]
            fid_index_dict[fid] = i - 1
            model_npy[i-1] = data[1:]
    mywritejson('../data/fid_index_dict.json',fid_index_dict)
    np.save('../data/model_50.npy',model_npy)

def select_patients():
    fid_index_dict = myreadjson('../data/fid_index_dict.json')
    patient_label_dict = dict()
    disease_index_dict = {
            'Obesity': 0,
            'Diabetes': 1,
            'COPD': 2,
            'HeartFailure': 3,
            }
    label_patient_dict = { k:[] for k in disease_index_dict.values() }
    for line in open('../data/sentence_sorted_by_date.csv'):
        data = line.strip().split()
        disease = data[0]
        pid = data[1]
        fid_list = data[2:]
        fid_list_useful = [fid for fid in fid_list if fid in fid_index_dict]
        if len(fid_list_useful) < 250 and len(fid_list_useful) > 50 :
            label = disease_index_dict[disease]
            if len(label_patient_dict[label]) < 3000:
                patient_label_dict[pid] = label
                label_patient_dict[label].append(pid)
    # for pid in patient_label_dict:
    mywritejson('../data/patient_label_dict.json',patient_label_dict)

def gen_data():
    patient_label_dict = myreadjson('../data/patient_label_dict.json')
    fid_index_dict = myreadjson('../data/fid_index_dict.json')
    patient_list = sorted(patient_label_dict.keys())

    
    x = np.ones([len(patient_label_dict),250], dtype=int) * 8627
    y = np.ones([len(patient_label_dict)], dtype=int)
    for line in open('../data/sentence_sorted_by_date.csv'):
        data = line.strip().split()
        fid_list = data[2:]
        pid = data[1]
        if pid in patient_label_dict:

            label = patient_label_dict[pid]

            pid_index = patient_list.index(pid)
            y[pid_index] = label
            x[pid_index][:len(fid_list)] = [fid_index_dict[fid] for fid in fid_list if fid in fid_index_dict]
    # print y
    # raw_input()
    # print x
    np.save('../data/x.npy',x)
    np.save('../data/y.npy',y)

def load_data():
    x = np.load('../data/x.npy')
    y = np.load('../data/y.npy')
    indices = range(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    train_num = int(0.7 * len(y))
    dev_num = int(0.8 * len(y))
    return x[:train_num],y[:train_num],x[train_num:dev_num],y[train_num:dev_num],x[dev_num:],y[dev_num:]

def load_data_all():
    x = np.load('../data/x.npy')
    y = np.load('../data/y.npy')
    return x,y



if __name__ == '__main__':
    gen_fid_index_dict()
    select_patients()
    gen_data()
    pass
