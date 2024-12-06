import os
import time
import csv
import sys
import dill
import random
import argparse
import operator
from tqdm import tqdm

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score     # CHANGE: new verion of library
from sklearn import metrics
from sklearn import tree
import sklearn.metrics as skm

import RF_fextract

# re-seed the generator
#np.random.seed(1234)

### Paths to data ###
# CHANGE: directory that will use when extracting feature
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data')
result_dir = os.path.join(current_dir, 'result')
alexa_monitored_data = f"{data_dir}/alexa"
hs_monitored_data = f"{data_dir}/hs_mon"
unmonitored_data = f"{data_dir}/unmon"

dic_of_feature_data = f"{data_dir}/tiktok.npz"          # the dataset that will use for feature extraction
dic_of_mon_data = f"{data_dir}/mon.pkl"                 # final feature that will use
dic_of_umon_data = f"{data_dir}/unmon.pkl"              # final feature that will use


### Parameters ###
# Number of sites, number of instances per site, number of (alexa/hs) monitored training instances per site, Number of trees for RF etc.
setting = "unmon_binary"    # mon, unmon_binary, unmon_multi

num_Trees = 1000

alexa_sites = 55        # train:test:validate = 8:1:1
alexa_instances = 800
alexa_train_inst = 640

mon_train_inst = alexa_train_inst
mon_test_inst = alexa_instances - mon_train_inst

hs_sites = 30
hs_instances = 100
hs_train_inst = 60

unmon_train = 20000
unmon_test = 10000                          # NOTE: change number of test to 10k, 40k, 70k
unmon_total = unmon_train + unmon_test      # CHANGE: total = train + test
unmon_test_str = "10k"

n_jobs = 64

threshold = None

# CHANGE: added log file
timestamp = time.strftime("%Y%m%d_%H%M%S")
file_name= f"{current_dir}/result/kfp_{setting}_{timestamp}.csv"
log_file = open(file_name, "w")


############ Feeder functions ############

def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):       # CHANGE: xrange to range
        yield l[i:i+n]


def checkequal(lst):
    return lst[1:] == lst[:-1]


############ Non-Feeder functions ########

def dictionary_(path_to_output_dict,
                path_to_input_dict = dic_of_feature_data,
                path_to_alexa = alexa_monitored_data, 
                path_to_hs = hs_monitored_data, 
                path_to_unmon = unmonitored_data,
                alexa_sites = alexa_sites,
                alexa_instances = alexa_instances, 
                hs_sites = hs_sites, 
                hs_instances = hs_instances, 
                unmon_sites = unmon_total
                ):
    '''Extract Features -- A dictionary containing features for each traffic instance.'''

    data_dict = {'alexa_feature': [],
                 'alexa_label': [],
                 'hs_feature': [],
                 'hs_label': [],
                 'unmonitored_feature': [],
                 'unmonitored_label': []}

    print("Creating Alexa features...")
    # CHANGE: changed to npz file format
    train_data = np.load(path_to_input_dict)
    x_train, y_train = train_data['data'], train_data['labels']

    for label in tqdm(range(alexa_sites)):
        instances = x_train[y_train == label]
        for idx, instance in enumerate(instances):
            g = []
            g.append(RF_fextract.TOTAL_FEATURES(instance))
            data_dict['alexa_feature'].append(g)
            data_dict['alexa_label'].append((label, idx))

    # orignal
    """
    for i in range(alexa_sites):
        for j in range(alexa_instances):
            fname = str(i) + "_" + str(j)
            if os.path.exists(path_to_alexa + fname):
                tcp_dump = open(path_to_alexa + fname).readlines()
                g = []
                g.append(RF_fextract.TOTAL_FEATURES(tcp_dump))
                data_dict['alexa_feature'].append(g)
                data_dict['alexa_label'].append((i,j))

    print("Creating HS features...")
    for i in tqdm(range(1, hs_sites + 1)):
        for j in range(hs_instances):
            fname = str(i) + "_" + str(j) + ".txt"
            if os.path.exists(path_to_hs + fname):
                tcp_dump = open(path_to_hs + fname).readlines()
                g = []
                g.append(RF_fextract.TOTAL_FEATURES(tcp_dump))
                data_dict['hs_feature'].append(g)
                data_dict['hs_label'].append((i,j))

    print("Creating Unmonitored features...")
    d, e = alexa_sites + 1, 0
    while e < unmon_sites:
        if e%500 == 0  and e>0:
            print(e)
        if os.path.exists(path_to_unmon + str(d)):
            tcp_dump = open(path_to_unmon + str(d)).readlines()
            g = []
            g.append(RF_fextract.TOTAL_FEATURES(tcp_dump))
            data_dict['unmonitored_feature'].append(g)
            data_dict['unmonitored_label'].append((d))
            d += 1
            e += 1
        else:
            d += 1
    """

    assert len(data_dict['alexa_feature']) == len(data_dict['alexa_label'])
    #assert len(data_dict['hs_feature']) == len(data_dict['hs_label'])
    #assert len(data_dict['unmonitored_feature']) == len(data_dict['unmonitored_label'])
    fileObject = open(path_to_output_dict, 'wb')
    dill.dump(data_dict, fileObject)
    fileObject.close()


def mon_train_test_references(mon_type, path_to_dict = dic_of_mon_data, setting = setting):
    """ Prepare monitored data in to training and test sets. """

    fileObject1 = open(path_to_dict, 'rb')      # CHANGE: 'r' to 'rb'
    dic = dill.load(fileObject1)

    if mon_type == 'alexa':
        split_data = list(chunks(dic['alexa_feature'], alexa_instances))
        split_target = list(chunks(dic['alexa_label'], alexa_instances))
    elif mon_type == 'hs':
        split_data = list(chunks(dic['hs_feature'], hs_instances))
        split_target = list(chunks(dic['hs_label'], hs_instances))

    training_data = []
    training_label = []
    test_data = []
    test_label = []

    for i in range(len(split_data)):
        temp = list(zip(split_data[i], split_target[i]))        # CHANGE: convert zip to list
        random.shuffle(temp)
        data, label = zip(*temp)
        training_data.extend(data[:mon_train_inst])
        training_label.extend(label[:mon_train_inst])
        test_data.extend(data[mon_train_inst:])
        test_label.extend(label[mon_train_inst:])

    flat_train_data = []
    flat_test_data = []

    # CHANGE: tuple () to list []
    for tr in training_data:
        flat_train_data.append(list(sum(tr, [])))
    for te in test_data:
        flat_test_data.append(list(sum(te, [])))

    # CHANGE: set all training labels to 0 in case of openworld binary
    if setting == "unmon_binary":
        training_label = [(0, t[1]) for t in training_label]
        test_label = [(0, t[1]) for t in test_label]

    # CHANGE: covert zip to list
    training_features =  list(zip(flat_train_data, training_label))
    test_features =  list(zip(flat_test_data, test_label))

    return training_features, test_features


def unmon_train_test_references(path_to_dict = dic_of_umon_data, setting = setting):
    """ Prepare unmonitored data in to training and test sets. """

    fileObject1 = open(path_to_dict, 'rb')      # CHANGE: 'r' to 'rb'
    dic = dill.load(fileObject1)

    training_data = []
    training_label = []
    test_data = []
    test_label = []

    unmon_data = dic['unmonitored_feature']

    # CHANGE: set label to 1 not 101 for binary classification
    if setting == "unmon_binary":
        unmon_label = [(1, i) for i in dic['unmonitored_label']]
    elif setting == "unmon_multi":
        unmon_label = [(101, i) for i in dic['unmonitored_label']]
    
    unmonitored = list(zip(unmon_data, unmon_label))        # CHANGE: convert zip to list, SHAPE: ([data, ...], (1, (-1, 0)))
    random.shuffle(unmonitored)
    u_data, u_label = zip(*unmonitored)     # SHAPE: u_data=[data, ...], u_label=(1, (-1, 0)))

    training_data.extend(u_data[:unmon_train])
    training_label.extend(u_label[:unmon_train])

    test_data.extend(u_data[unmon_train:unmon_train+unmon_test])    # CHANGE: unmon_total to unmon_train+unmon_test
    test_label.extend(u_label[unmon_train:unmon_train+unmon_test])  # CHANGE: unmon_total to unmon_train+unmon_test

    flat_train_data = []
    flat_test_data = []

    # CHANGE: tuple () to list []
    for tr in training_data:
        flat_train_data.append(list(sum(tr, [])))
    for te in test_data:
        flat_test_data.append(list(sum(te, [])))

    # CHANGE: covert zip to list
    training_features =  list(zip(flat_train_data, training_label))
    test_features =  list(zip(flat_test_data, test_label))

    return training_features, test_features


def RF_closedworld(mon_type, path_to_dict = dic_of_mon_data, n_jobs = n_jobs):
    '''Closed world RF classification of data -- only uses sk.learn classification - does not do additional k-nn.'''

    training, test = mon_train_test_references(mon_type, path_to_dict)
    tr_data, tr_label1 = zip(*training)

    # CHANGE: tuple to list
    tr_label = list(zip(*tr_label1))[0]
    te_data, te_label1 = zip(*test)
    te_label = list(zip(*te_label1))[0]

    print("Monitored type: ", mon_type)
    print("Training...")

    model = RandomForestClassifier(n_jobs=n_jobs, n_estimators=num_Trees, oob_score=True, verbose=1)    # CHANGE: added verbose
    model.fit(tr_data, tr_label)
    
    print("RF accuracy = ", model.score(te_data, te_label))
    print("Feature importance scores:", model.feature_importances_)

    scores = cross_val_score(model, np.array(tr_data), np.array(tr_label))
    print("cross_val_score = ", scores.mean())
    print("OOB score = ", model.oob_score_)


def RF_openworld(mon_type, path_to_mon_dict = dic_of_mon_data, path_to_unmon_dict = dic_of_umon_data, n_jobs=n_jobs):
    '''Produces leaf vectors used for classification.'''

    # get monitored and unmonitored data
    mon_training, mon_test = mon_train_test_references(mon_type, path_to_mon_dict)
    unmon_training, unmon_test = unmon_train_test_references(dic_of_umon_data)

    # add mon and unmon data
    training = mon_training + unmon_training
    test = mon_test + unmon_test

    # separate feature and label (bring label from (0, 936) of (1, (-1, 0)) shape)
    tr_data, tr_label1 = zip(*training)
    tr_label = list(zip(*tr_label1))[0]     # CHANGE: convert zip to list
    te_data, te_label1 = zip(*test)
    te_label = list(zip(*te_label1))[0]     # CHANGE: convert zip to list

    print(f"train/test label length: {len(tr_label)}, {len(te_label)}")

    print("Training...")
    model = RandomForestClassifier(n_jobs=n_jobs, n_estimators=num_Trees, oob_score=True, verbose=1)    # CHANGE: added verbose
    model.fit(tr_data, tr_label)

    # make leaf vector in train/test data
    train_leaf = zip(model.apply(tr_data), tr_label)
    test_leaf = zip(model.apply(te_data), te_label)
    return train_leaf, test_leaf


def distances(mon_type, path_to_mon_dict = dic_of_mon_data, path_to_unmon_dict = dic_of_umon_data, unmon_test_str = unmon_test_str, keep_top=100):
    """ This uses the above function to calculate distance from test instance between each training instance (which are used as labels) and writes to file
        Default keeps the top 100 instances closest to the instance we are testing.
        -- Saves as (distance, true_label, predicted_label) --
    """
    # do experiments
    train_leaf, test_leaf = RF_openworld(mon_type, path_to_mon_dict, path_to_unmon_dict)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    monitored_directory = f"{result_dir}/monitored_{mon_type}/"         # CHANGE: changed directory name
    if not os.path.exists(monitored_directory):
        os.mkdir(monitored_directory)
    unmonitored_directory = f"{result_dir}/unmonitored_{mon_type}_{unmon_test_str}/"
    if not os.path.exists(unmonitored_directory):
        os.mkdir(unmonitored_directory)

    # Make into numpy arrays
    train_leaf = [(np.array(l, dtype=int), v) for l, v in train_leaf]
    test_leaf = [(np.array(l, dtype=int), v) for l, v in test_leaf]

    if mon_type == 'alexa':
        sites = alexa_sites
    elif mon_type == 'hs':
        sites = hs_sites

    print("Calculating mon test leaf...")
    for i, instance in enumerate(test_leaf[:(mon_test_inst*sites)]):
        if i%100==0:
            print("%d out of %d" % (i, mon_test_inst * sites))                  # CHANGE: stdout to print
        temp = []
        for item in train_leaf:
            # vectorize the average distance computation
            d = np.sum(item[0] != instance[0]) / float(item[0].size)
            if d == 1.0:
                continue
            temp.append((d, instance[1], item[1]))
        tops = sorted(temp)[:keep_top]
        myfile = open(monitored_directory  + '%d_%s.txt' %(instance[1], i), 'w')
        for item in tops:
            myfile.write("%s\n" % str(item))
        myfile.close()
    print("Calculated mon test leaf")

    print("Calculating unmon test leaf...")
    for i, instance in enumerate(test_leaf[(mon_test_inst*sites):]):
        if i%100==0:
            print("%d out of %d" % (i, len(test_leaf) - mon_test_inst * sites))   # CHANGE: stdout to print
        temp = []
        for item in train_leaf:
            # vectorize the average hamming distance computation
            d = np.sum(item[0] != instance[0]) / float(item[0].size)
            if d == 1.0:
                continue
            temp.append((d, instance[1], item[1]))
        tops = sorted(temp)[:keep_top]
        myfile = open(unmonitored_directory  + '%d_%s.txt' %(instance[1], i), 'w')
        for item in tops:
            myfile.write("%s\n" % str(item))
        myfile.close()
    print("Calculated unmon test leaf")


def distance_stats(mon_type, result_dir, knn=3, threshold=threshold):
    """ For each test instance this picks out the minimum training instance distance, 
        checks (for mon) if it is the right label and checks if it's knn are the same label
    """

    monitored_directory = f"{result_dir}/monitored_{mon_type}/"                         # CHANGE: changed directory name
    unmonitored_directory = f"{result_dir}/unmonitored_{mon_type}_{unmon_test_str}/"

    TP = 0
    FN = 0
    FP = 0
    TN = 0
    monitored_label = 0.0
    unmonitored_label  = 1.0

    # original
    if threshold is None:
        for subdir, dirs, files in os.walk(monitored_directory):
            for file in files:
                fn = os.path.join(subdir, file)
                data = open(str(fn)).readlines()
                internal_count = 0
                for i in data[:knn]:
                    distance = float(eval(i)[0])
                    true_label = float(eval(i)[1])
                    predicted_label = float(eval(i)[2])
                    if true_label == predicted_label:
                        internal_count += 1
                if internal_count == knn:
                    TP += 1
        
        path, dirs, files = next(os.walk(monitored_directory))
        file_count1 = len(files)
        print("TP = ", TP/float(file_count1))
        
        for subdir, dirs, files in os.walk(unmonitored_directory):
            for file in files:
                fn = os.path.join(subdir, file)
                data = open(str(fn)).readlines()
                internal_count = 0
                test_list = []
                internal_test = []
                for i in data[:knn]:
                    distance = float(eval(i)[0])
                    true_label = float(eval(i)[1])
                    predicted_label = float(eval(i)[2])
                    internal_test.append(predicted_label)
                if checkequal(internal_test) == True and internal_test[0] <= alexa_sites:
                    FP += 1

        path, dirs, files = next(os.walk(unmonitored_directory))
        file_count2 = len(files)
        print("FP = ", FP/float(file_count2))

        log_file.writelines("%.6f,%6f\n"%(TP/float(file_count1), FP/float(file_count2)))
    # endif


    # CHANGE: when there is threshold
    else:
        for subdir, dirs, files in os.walk(monitored_directory):
            for file in files:
                fn = os.path.join(subdir, file)
                data = open(str(fn)).readlines()
                internal_count = 0
                for i in data[:knn]:
                    distance = float(eval(i)[0])
                    true_label = float(eval(i)[1])
                    predicted_label = float(eval(i)[2])
                    
                    # CHANGE: added more matric
                    if predicted_label == monitored_label:
                        if distance <= threshold:           # predicted as Monitored and actual site is Unmonitored
                            FP = FP + 1
                        else:                               # predicted as Unmonitored and actual site is Unmonitored
                            TN = TN + 1
                    elif predicted_label == unmonitored_label:
                        TN = TN + 1                         # predicted as Unmonitored and actual site is Unmonitored

        for subdir, dirs, files in os.walk(unmonitored_directory):
            for file in files:
                fn = os.path.join(subdir, file)
                data = open(str(fn)).readlines()
                internal_count = 0
                for i in data[:knn]:
                    distance = float(eval(i)[0])
                    true_label = float(eval(i)[1])
                    predicted_label = float(eval(i)[2])
                
                # CHANGE: added more matric
                if predicted_label == monitored_label:
                    if distance <= threshold:               # predicted as Monitored and actual site is Unmonitored
                        FP = FP + 1
                    else:                                   # predicted as Unmonitored and actual site is Unmonitored
                        TN = TN + 1
                elif predicted_label == unmonitored_label:
                    TN = TN + 1                             # predicted as Unmonitored and actual site is Unmonitored

        print ("TP : ", TP)
        print ("FP : ", FP)
        print ("TN : ", TN)
        print ("FN : ", FN)
        print ("Total  : ", TP + FP + TN + FN)
        TPR = float(TP) / (TP + FN)
        print ("TPR =", TPR)
        FPR = float(FP) / (FP + TN)
        print ("FPR =",  FPR)
        Precision = float(TP) / (TP + FP)
        print ("Precision =", Precision)
        Recall = float(TP) / (TP + FN)
        print ("Recall =", Recall)
        log_file.writelines("%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n"%(threshold_val, TP, FP, TN, FN, TPR, FPR, Precision, Recall))
    # endelse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='k-FP benchmarks')
    parser.add_argument('--dictionary', action='store_true', help='Build dictionary.')
    parser.add_argument('--RF_closedworld', action='store_true', help='Closed world classification.')
    parser.add_argument('--distances', action='store_true', help='Build distances for open world classification.')
    parser.add_argument('--distance_stats', action='store_true', help='Open world classification.')
    parser.add_argument('--knn', nargs=1, metavar="INT", help='Number of nearest neighbours.')
    parser.add_argument('--mon_type', nargs=1, metavar="STR", help='The type of monitored dataset - alexa or hs.')
    args = parser.parse_args()

    # feature extraction
    if args.dictionary:
        # Example command line:
        # $ python3 k-FP.py --dictionary
        dictionary_(path_to_output_dict=dic_of_mon_data)

    # closed world experiments
    elif args.RF_closedworld:
        # Example command line:
        # $ python3 k-FP.py --RF_closedworld --mon_type alexa
        mon_type = str(args.mon_type[0])
        RF_closedworld(mon_type)

    # open world experiments
    elif args.distances:
        # Example command line:
        # $ python3 k-FP.py --distances --mon_type alexa
        mon_type = str(args.mon_type[0])
        distances(mon_type)

    # evaluation
    elif args.distance_stats:
        # Example command line:
        # $ python3 k-FP.py --distance_stats --knn 6 --mon_type alexa
        knn = int(args.knn[0])
        mon_type = str(args.mon_type[0])
        distance_stats(mon_type, result_dir, knn)