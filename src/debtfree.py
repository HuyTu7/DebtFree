from __future__ import division, print_function

from datetime import timedelta
import pandas as pd
import numpy as np
import sklearn
import random
import pdb
from demos import cmd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from os import listdir

import collections

try:
    import cPickle as pickle
except:
    import pickle
from learners import Treatment, TM, SVM, RF, DT, NB, LR
from clahard import *
from jitterbug import *
import warnings

warnings.filterwarnings('ignore')

BUDGET = 50
POOL_SIZE = 10000
INIT_POOL_SIZE = 10
np.random.seed(4789)


def load_csv(path="../new_data/original/"):
    data = {}
    for file in listdir(path):
        if ".csv" in file:
            try:
                df = pd.read_csv(path + file)
                data[file.split(".csv")[0]] = df
            except:
                print("Ill-formated file", file)
    return data


def getHigherValueCutoffs(data, percentileCutoff, class_category):
    '''
	Parameters
	----------
	data : in pandas format
	percentileCutoff : in integer
	class_category : [TODO] not needed

	Returns
	-------
	'''
    abc = data.quantile(float(percentileCutoff) / 100)
    abc = np.array(abc.values)[:-1]
    return abc


def filter_row_by_value(row, cutoffsForHigherValuesOfAttribute):
    '''
	Shortcut to filter by rows in pandas
	sum all the attribute values that is higher than the cutoff
	----------
	row
	cutoffsForHigherValuesOfAttribute

	Returns
	-------
	'''
    rr = row[:-1]
    condition = np.greater(rr, cutoffsForHigherValuesOfAttribute)
    res = np.count_nonzero(condition)
    return res


def getInstancesByCLA(data, percentileCutOff, positiveLabel):
    '''
	- unsupervised clustering by median per attribute
	----------
	data
	percentileCutOff
	positiveLabel

	Returns
	-------

	'''
    # get cutoff per fixed percentile for all the attributes
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(data, percentileCutOff, "Label")
    # get K for all the rows
    K = data.apply(lambda row: filter_row_by_value(row, cutoffsForHigherValuesOfAttribute), axis=1)
    # cutoff for the cluster to be partitioned into
    cutoffOfKForTopClusters = np.percentile(K, percentileCutOff)
    instances = [1 if x > cutoffOfKForTopClusters else 0 for x in K]
    data["CLA"] = instances
    data["K"] = K
    return data


def getInstancesByRemovingSpecificAttributes(data, attributeIndices, invertSelection, label="Label"):
    '''
	removing the attributes
	----------
	data
	attributeIndices
	invertSelection

	Returns
	-------
	'''
    if not invertSelection:
        data_res = data.drop(data.columns[attributeIndices], axis=1)
    else:
        # invertedIndices = np.in1d(range(len(attributeIndices)), attributeIndices)
        # data.drop(data.columns[invertedIndices], axis=1, inplace=True)
        data_res = data[attributeIndices]
        data_res['Label'] = data[label].values
    return data_res


def getInstancesByRemovingSpecificInstances(data, instanceIndices, invertSelection):
    '''
	removing instances
	----------
	data
	instanceIndices
	invertSelection

	Returns
	-------

	'''
    if not invertSelection:
        data.drop(instanceIndices, axis=0, inplace=True)
    else:
        invertedIndices = np.in1d(range(data.shape[0]), instanceIndices)
        data.drop(invertedIndices, axis=0, inplace=True)
    return data


def getSelectedInstances(data, cutoffsForHigherValuesOfAttribute, positiveLabel):
    '''
	select the instances that violate the assumption
	----------
	data
	cutoffsForHigherValuesOfAttribute
	positiveLabel

	Returns
	-------
	'''
    violations = data.apply(lambda r: getViolationScores(r,
                                                         data['Label'],
                                                         cutoffsForHigherValuesOfAttribute),
                            axis=1)
    violations = violations.values
    # get indices of the violated instances
    selectedInstances = (violations > 0).nonzero()[0]
    # remove randomly 90% of the instances that violate the assumptions
    # selectedInstances = np.random.choice(selectedInstances, int(selectedInstances.shape[0] * 0.9), replace=False)
    # for index in range(data.shape[0]):
    # 	if violations[index] > 0:
    # 		selectedInstances.append(index)
    tmp = data.iloc[selectedInstances]
    if tmp[tmp["Label"] == 1].shape[0] < 10:
        print("not enough data after removing instances")
        len_0 = selectedInstances.shape[0]
        len_0 -= data[data["Label"] == 1].shape[0]
        selectedInstances = np.random.choice(selectedInstances, int(len_0), replace=False)

    return selectedInstances


def rq1(seed=0, input="../new_data/corrected/", output="../results/SE_CLAMI_90_"):
    '''la90
	main method for most of the methods:
	- CLA
	- CLAMI
	- FLASH_CLAMI
	----------
	seed
	input
	output

	Returns
	-------
	'''
    # treatments = ["CLA", "FLASH_CLA", "CLA+RF", "CLA+NB"]
    treatments = ["CLA", "CLAMI+RF", "CLAMI+NB", "CLA+RF"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())
    result = {}
    keys = list(data.keys())
    keys.sort()
    print(keys)
    for target in keys:
        print(target)
        result[target] = [CLA(data, target, None, 50)]
        # result[target] += [tune_CLAMI(data, target, None, 50)]
        # print(result[target][-1])
        result[target] += CLAMI(data, target, None, 50)
        result[target].append(CLA_SL(data, target, model="RF", seed=seed))
        print(result[target])
    result["Treatment"] = treatments
    # Output results to tables
    metrics = result[columns[-1]][0].keys()
    for metric in metrics:
        df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in result}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)


def two_step_CLAHARD(data, target, model="RF", est=False, T_rec=0.9, inc=False, seed=0,
                       hybrid=True, CLAfilter=False, CLAlabel=True, toggle=False, early_stop=False):
    np.random.seed(seed)
    jitterbug = CLAHard(data, target, cla=False, thres=55, early_stop=early_stop,
                        CLAfilter=CLAfilter, CLAlabel=CLAlabel, hybrid=hybrid, toggle=toggle)
    # jitterbug.find_patterns()
    jitterbug.test_patterns(include=inc)
    tmp = jitterbug.rest[target]
    print(len(tmp[tmp["label"] == "yes"]))

    jitterbug.ML_hard(model=model, est=est, T_rec=T_rec)
    # stats = None
    return jitterbug


def ECLA(data, target, model="RF", est=False, T_rec=0.90, inc=False, seed=0, both=False):
    jitterbug = Jitterbug(data, target)
    jitterbug.find_patterns()
    jitterbug.easy_code()
    jitterbug.test_patterns()
    rest_data = jitterbug.rest
    treatment = Treatment(rest_data, target)
    treatment.preprocess()

    test_data = treatment.full_test
    if both:
        test_data = [test_data, treatment.full_train]
        test_data = pd.concat(test_data, ignore_index=True)
    final_data = getInstancesByCLA(test_data, 90, None)
    final_data = final_data[:treatment.full_test.shape[0]]
    treatment.y_label = ["yes" if y == 1 else "no" for y in final_data["Label"]]
    treatment.decisions = ["yes" if y == 1 else "no" for y in final_data["CLA"]]
    treatment.probs = final_data["K"]
    treatment.stats = jitterbug.easy.stats_test

    return treatment, rest_data


def CLA_SL(data, target, model="RF", est=False, T_rec=0.90, inc=False, seed=0, both=False, stats={"tp": 0, "p": 0}):
    tm = {"RF": RF, "SVM": SVM, "LR": LR, "NB": NB, "DT": DT, "TM": TM}

    treatment = Treatment(data, target)
    treatment.preprocess()
    traindata = treatment.full_train
    full_data = getInstancesByCLA(traindata, 90, None)

    tm = tm[model]
    clf = tm(data, target)
    print(target, model)
    clf.preprocess()
    clf.x_label = ["yes" if x == 1 else "no" for x in full_data['CLA']]
    clf.train()
    clf.stats = stats
    results = clf.eval()
    return results


def get_CLASUP(seed=0, input="../new_data/corrected/", output="../results/SE_CLA90+SUP_"):
    treatments = ["LR", "DT", "RF", "SVM", "NB"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())

    # Supervised Learning Results
    result = {}
    for target in data:
        result[target] = [CLA_SL(data, target, model=model, seed=seed) for model in treatments]
        # Output results to tables
        metrics = result[target][0].keys()
        print(result[target])
        for metric in metrics:
            df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in
                  result}
            pd.DataFrame(df, columns=columns).to_csv(output + metric + ".csv", line_terminator="\r\n",
                                                     index=False)




def get_EASYCLA(seed=0, input="../new_data/corrected/", output="../results/SE_EASYCLA_orig_"):
    treatments = ["E_CLA", "E_CLAMI_RF", "E_CLAMI_NB"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())
    result = {}
    keys = list(data.keys())
    keys.sort()
    print(keys)
    for target in keys:
        tm, rest_data = ECLA(data, target, est=False, inc=False, seed=seed, both=False)
        result[target] = [tm.eval()]
        result[target] += CLAMI(rest_data, target, None, 50, stats=tm.stats)

        print(target)
        print(result[target])

    result["Treatment"] = treatments
    # Output results to tables
    metrics = result[columns[-1]][0].keys()
    for metric in metrics:
        df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in result}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)
    data_type = "SE_EASYCLA_unsup"
    with open("../dump/%s_result.pickle" % data_type, "wb") as f:
        pickle.dump(result, f)


def unlabelled_data(seed=0, input="../new_data/corrected/", output="../results/SE_"):
    treatments = ["L_Hard", "L_F_Hard", "L_Falcon", "L_F_Falcon"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())
    result = {}
    keys = list(data.keys())
    keys.sort()
    data_type = output.split("/")[-1]
    print(keys, data_type, "../dump/%sresult.pickle" % data_type)
    total_yes_count = 0
    yes_count_median = []
    stats_satds = {}
    keys = ["apache-jmeter-2.10", "jruby-1.4.0", "hibernate-distribution-3.3.2.GA",
            "emf-2.4.1", "apache-ant-1.7.0", "sql12",
            "columba-1.4-src", "argouml", "jfreechart-1.0.19", "jEdit-4.2"]
    for target in keys:
        # Labeling (CLA) + Hard
        L_hard = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                       hybrid=True, CLAfilter=False, CLAlabel=True, toggle=False, early_stop=False)
        result[target] = [L_hard.eval()]
        # Labeling (CLA) + Filtering (CLA) + Hard
        L_F_hard = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                     hybrid=False, CLAfilter=True, CLAlabel=True, toggle=False, early_stop=False)
        result[target].append(L_F_hard.eval())
        # Labeling (CLA) + Falcon
        L_falcon = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                    hybrid=True, CLAfilter=False, CLAlabel=True, toggle=False, early_stop=False)
        result[target].append(L_falcon.eval())
        # Labeling (CLA) + Filtering (CLA) + Falcon
        L_F_falcon = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                    hybrid=True, CLAfilter=True, CLAlabel=True, toggle=False, early_stop=False)
        result[target].append(L_F_falcon.eval())
        stats_satds[target] = L_F_falcon.stats_satd
        print(result[target])

    with open("../dump/%sSATD_stats.pickle" % data_type, "wb") as f:
        pickle.dump(stats_satds, f)
    print(total_yes_count, np.median(yes_count_median), np.median(yes_count_median) / total_yes_count)

    result["Treatment"] = treatments
    # Output results to tables
    metrics = result[columns[-1]][0].keys()
    for metric in metrics:
        df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in keys}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)

    with open("../dump/%sresult.pickle" % data_type, "wb") as f:
        pickle.dump(result, f)


def labelled_data(seed=0, input="../new_data/corrected/", output="../results/SE_"):
    treatments = ["L_Hard", "L_F_Hard", "L_Falcon", "L_F_Falcon"]
    data = load_csv(path=input)
    columns = ["Treatment"] + list(data.keys())
    result = {}
    keys = list(data.keys())
    keys.sort()
    data_type = output.split("/")[-1]
    print(keys, data_type, "../dump/%sresult.pickle" % data_type)
    total_yes_count = 0
    yes_count_median = []
    stats_satds = {}
    keys = ["apache-jmeter-2.10", "jruby-1.4.0", "hibernate-distribution-3.3.2.GA",
            "emf-2.4.1", "apache-ant-1.7.0", "sql12",
            "columba-1.4-src", "argouml", "jfreechart-1.0.19", "jEdit-4.2"]
    for target in keys:
        # Hard
        L_hard = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                       hybrid=False, CLAfilter=False, CLAlabel=False, toggle=False, early_stop=False)
        result[target] = [L_hard.eval()]
        # Filtering (CLA) + Hard
        L_F_hard = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                     hybrid=False, CLAfilter=True, CLAlabel=False, toggle=False, early_stop=False)
        result[target].append(L_F_hard.eval())
        # Falcon
        L_falcon = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                    hybrid=True, CLAfilter=False, CLAlabel=False, toggle=False, early_stop=False)
        result[target].append(L_falcon.eval())
        # Filtering (CLA) + Falcon
        L_F_falcon = two_step_CLAHARD(data, target, est=True, inc=False, T_rec=0.9, seed=seed,
                                    hybrid=True, CLAfilter=True, CLAlabel=False, toggle=False, early_stop=False)
        result[target].append(L_F_falcon.eval())
        stats_satds[target] = L_F_falcon.stats_satd
        print(result[target])

    with open("../dump/%sSATD_stats.pickle" % data_type, "wb") as f:
        pickle.dump(stats_satds, f)
    print(total_yes_count, np.median(yes_count_median), np.median(yes_count_median) / total_yes_count)

    result["Treatment"] = treatments
    # Output results to tables
    metrics = result[columns[-1]][0].keys()
    for metric in metrics:
        df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in keys}
        pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
                                                 line_terminator="\r\n", index=False)

    with open("../dump/%sresult.pickle" % data_type, "wb") as f:
        pickle.dump(result, f)


def CLA(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, both=False):
    treatment = Treatment(data, target)
    treatment.preprocess()
    testdata = treatment.full_test
    data = getInstancesByCLA(testdata, percentileCutoff, positiveLabel)
    treatment.y_label = ["yes" if y == 1 else "no" for y in data["Label"]]
    treatment.decisions = ["yes" if y == 1 else "no" for y in data["CLA"]]
    treatment.probs = data["K"]
    return treatment.eval()


def CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, stats={"tp": 0, "p": 0},
          label="Label"):
    '''
	CLAMI - Clustering, Labeling, Metric/Features Selection,
			Instance selection, and Supervised Learning
	----------

	Returns
	-------

	'''
    treatment = Treatment(data, target)
    treatment.preprocess()
    data = treatment.full_train
    testdata = treatment.full_test
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(data, percentileCutoff, "Label")
    print("get cutoffs")
    data = getInstancesByCLA(data, percentileCutoff, positiveLabel)
    print("get CLA instances")

    metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(data,
                                                                                 cutoffsForHigherValuesOfAttribute,
                                                                                 positiveLabel, label=label)
    print("get Features and the violation scores")
    # pdb.set_trace()
    keys = list(metricIdxWithTheSameViolationScores.keys())
    # start with the features that have the lowest violation scores
    keys.sort()
    for i in range(len(keys)):
        k = keys[i]
        selectedMetricIndices = metricIdxWithTheSameViolationScores[k]
        # while len(selectedMetricIndices) < 3:
        # 	index = i + 1
        # 	selectedMetricIndices += metricIdxWithTheSameViolationScores[keys[index]]
        print(selectedMetricIndices)
        # pick those features for both train and test sets
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
                                                                            selectedMetricIndices, True, label=label)
        newTestInstances = getInstancesByRemovingSpecificAttributes(testdata,
                                                                    selectedMetricIndices, True, label="Label")
        # restart looking for the cutoffs in the train set
        cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                                  percentileCutoff, "Label")
        # get instaces that violated the assumption in the train set
        instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                       cutoffsForHigherValuesOfAttribute,
                                                       positiveLabel)
        # remove the violated instances
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                           instIndicesNeedToRemove, False)

        # make sure that there are both classes data in the training set
        zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
        one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
        if zero_count > 0 and one_count > 0:
            break

    return CLAMI_eval(trainingInstancesByCLAMI, newTestInstances, target, stats=stats)


def CLAMI_data(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, stats={"tp": 0, "p": 0},
               label="Label"):
    '''
	CLAMI - Clustering, Labeling, Metric/Features Selection,
			Instance selection, and Supervised Learning
	----------

	Returns
	-------

	'''
    treatment = Treatment(data, target)
    treatment.preprocess()
    data = treatment.full_train
    testdata = treatment.full_test
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(data, percentileCutoff, "Label")
    print("get cutoffs")
    data = getInstancesByCLA(data, percentileCutoff, positiveLabel)
    print("get CLA instances")

    metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(data,
                                                                                 cutoffsForHigherValuesOfAttribute,
                                                                                 positiveLabel, label=label)
    print("get Features and the violation scores")
    # pdb.set_trace()
    keys = list(metricIdxWithTheSameViolationScores.keys())
    # start with the features that have the lowest violation scores
    keys.sort()
    for i in range(len(keys)):
        k = keys[i]
        selectedMetricIndices = metricIdxWithTheSameViolationScores[k]
        # while len(selectedMetricIndices) < 3:
        # 	index = i + 1
        # 	selectedMetricIndices += metricIdxWithTheSameViolationScores[keys[index]]
        print(selectedMetricIndices)
        # pick those features for both train and test sets
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
                                                                            selectedMetricIndices, True, label=label)
        newTestInstances = getInstancesByRemovingSpecificAttributes(testdata,
                                                                    selectedMetricIndices, True, label="Label")
        # restart looking for the cutoffs in the train set
        cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                                  percentileCutoff, "Label")
        # get instaces that violated the assumption in the train set
        instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                       cutoffsForHigherValuesOfAttribute,
                                                       positiveLabel)
        # remove the violated instances
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                           instIndicesNeedToRemove, False)

        # make sure that there are both classes data in the training set
        zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
        one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
        if zero_count > 0 and one_count > 0:
            break

    return trainingInstancesByCLAMI, newTestInstances


def CLAMI_eval(trainingInstancesByCLAMI, newTestInstances, target, stats={"tp": 0, "p": 0}):
    results = []
    # treaments = ["LR", "SVM", "RF", "NB"]
    treaments = ["RF", "NB"]
    for mlAlg in treaments:
        results.append(training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, mlAlg, stats=stats))
    return results


def MI(data, tunedata, selectedMetricIndices, percentileCutoff, positiveLabel, target):
    print(selectedMetricIndices)
    trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
                                                                        selectedMetricIndices, True, label="CLA")
    newTuneInstances = getInstancesByRemovingSpecificAttributes(tunedata,
                                                                selectedMetricIndices, True, label="Label")
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                              percentileCutoff, "Label")
    instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                   cutoffsForHigherValuesOfAttribute,
                                                   positiveLabel)
    trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                       instIndicesNeedToRemove, False)
    zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
    one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
    if zero_count > 0 and one_count > 0:
        return selectedMetricIndices, training_CLAMI(trainingInstancesByCLAMI, newTuneInstances, target, "RF")
    else:
        return -1, -1



def training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, model, all=True, stats={"tp": 0, "p": 0}):
    treatments = {"RF": RF, "SVM": SVM, "LR": LR, "NB": NB, "DT": DT, "TM": TM}
    treatment = treatments[model]
    clf = treatment(trainingInstancesByCLAMI, target)
    print(target, model)
    clf.test_data = newTestInstances[newTestInstances.columns.difference(['Label'])].values
    clf.y_label = np.array(["yes" if x == 1 else "no" for x in newTestInstances["Label"].values])

    try:
        clf.train_data = trainingInstancesByCLAMI.values[:, :-1]
        clf.x_label = np.array(["yes" if x == 1 else "no" for x in trainingInstancesByCLAMI['Label']])
        clf.train()
        clf.stats = stats
        results = clf.eval()
        if all:
            return results
        else:
            return results["APFD"] + results["f1"]
    except:
        pdb.set_trace()


def getViolationScores(data, labels, cutoffsForHigherValuesOfAttribute, key=-1):
    '''
	get violation scores
	----------
	data
	labels
	cutoffsForHigherValuesOfAttribute
	key

	Returns
	-------

	'''
    violation_score = 0
    if key not in ["Label", "K", "CLA"]:
        if key != -1:
            # violation score by columns
            categories = labels.values
            cutoff = cutoffsForHigherValuesOfAttribute[key]
            # violation: less than a median and class = 1 or vice-versa
            violation_score += np.count_nonzero(np.logical_and(categories == 0, np.greater(data.values, cutoff)))
            violation_score += np.count_nonzero(np.logical_and(categories == 1, np.less_equal(data.values, cutoff)))
        else:
            # violation score by rows
            row = data.values
            row_data, row_label = row[:-1], row[-1]
            # violation: less than a median and class = 1 or vice-versa

            row_label_0 = np.array(row_label == 0).tolist() * row_data.shape[0]
            # randomness = random.random()
            # if randomness > 0.5:
            violation_score += np.count_nonzero(np.logical_and(row_label_0,
                                                               np.greater(row_data, cutoffsForHigherValuesOfAttribute)))
            row_label_1 = np.array(row_label == 0).tolist() * row_data.shape[0]
            violation_score += np.count_nonzero(np.logical_and(row_label_1,
                                                               np.less_equal(row_data,
                                                                             cutoffsForHigherValuesOfAttribute)))

    # for attrIdx in range(data.shape[1] - 3):
    # 	# if attrIdx not in ["Label", "CLA"]:
    # 	attr_data = data[attrIdx].values
    # 	cutoff = cutoffsForHigherValuesOfAttribute[attrIdx]
    # 	violations.append(getViolationScoreByColumn(attr_data, data["Label"], cutoff))
    return violation_score


def getMetricIndicesWithTheViolationScores(data, cutoffsForHigherValuesOfAttribute, positiveLabel, label="Label"):
    '''
	get all the features that violated the assumption
	----------
	data
	cutoffsForHigherValuesOfAttribute
	positiveLabel

	Returns
	-------

	'''
    # cutoffs for all the columns/features
    cutoffsForHigherValuesOfAttribute = {i: x for i, x in enumerate(cutoffsForHigherValuesOfAttribute)}
    # use pandas apply per column to find the violation scores of all the features
    violations = data.apply(
        lambda col: getViolationScores(col, data[label], cutoffsForHigherValuesOfAttribute, key=col.name),
        axis=0)
    violations = violations.values
    metricIndicesWithTheSameViolationScores = collections.defaultdict(list)

    # store the violated features that share the same violation scores together
    for attrIdx in range(data.shape[1] - 3):
        key = violations[attrIdx]
        metricIndicesWithTheSameViolationScores[key].append(attrIdx)
    return metricIndicesWithTheSameViolationScores


if __name__ == "__main__":
    eval(cmd())
