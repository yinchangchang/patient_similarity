## Measuring Patient Similarities via a Deep Architecture with Medical Concept Embedding; Official TensorFlow Implementation
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg?style=plastic)
![TensorFlow 0.12.0](https://img.shields.io/badge/tensorflow-0.12.0-green.svg?style=plastic)


This repository contains the official TensorFlow implementation of the following paper:

> **Measuring Patient Similarities via a Deep Architecture with Medical Concept Embedding**<br>
> Zihao Zhu, Changchang Yin, Buyue Qian, Yu Cheng, Jishang Wei, Fei Wang<br>
> [paper](https://ieeexplore.ieee.org/document/7837899)
>
> **Abstract:** *Evaluating the clinical similarities between pairwise patients is a fundamental problem in healthcare informatics. Aproper patient similarity measure enables various downstream applications, such as cohort study and treatment comparative effectiveness research. One major carrier for conducting patient similarity research is the Electronic Health Records(EHRs), which are usually heterogeneous, longitudinal, and sparse. Though existing studies on learning patient similarity from EHRs have shown being useful in solving real clinical problems, their applicability is limited due to the lack of medical interpretations. Moreover, most previous methods assume a vector based representation for patients, which typically requires aggregation of medical events over a certain time period. As aconsequence, the temporal information will be lost. In this paper, we propose a patient similarity evaluation framework based on temporal matching of longitudinal patient EHRs. Two efficient methods are presented, unsupervised and supervised, both of which preserve the temporal properties in EHRs. The supervised scheme takes a convolutional neural network architecture, and learns an optimal representation of patient clinical records with medical concept embedding. The empirical results on real-world clinical data demonstrate substantial improvement over the baselines.*


## Training networks

You can train the networks as follows:

1. Edit [model_50.vector](data/model_50.vector) and [sentence_sorted_by_date.csv](data/sentence_sorted_by_date.csv) according to your own EHR data.
2. cd code
3. python data_helper.py
4. python train.py
5. python test.py

## files

### model_50.vector
Embedding file: The medical events appeared in [sentence_sorted_by_date.csv](data/sentence_sorted_by_date.csv) and their embeddings.


### sentence_sorted_by_date.csv
-	Each line of the file contains:
	1.	cohort name
	2.	patient id
	3.	medical event sequences
-	For each patient, by concatenating all medical events in his/her EHRs according to their happening timestamps (for events with the same timestamp we do not care about the order), we obtained a sequences.
