#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:35:53 2019

@author: wangxiujiang
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_iris
from alipy.experiment.al_experiment import AlExperiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import json
import time
import pandas as pd
import numpy as np
import multiprocessing
import warnings
import argparse
import jieba

#X, y = load_iris(return_X_y=True)

# new models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
#import xgboost as xgb

"""
import argparse

parser = argparse.ArgumentParser(description="preprocess data.csv to train_data and test_data, weibo.csv to data_weibo.csv ...")
parser.add_argument("-d", "--dataset", default='20newsgroups', help="20newsgroups sms sougou quora jigsaw")
args = parser.parse_args()
print(args.dataset)

"""

models = [
#            MultinomialNB(alpha=.01), #very fast ~0.7
            LogisticRegression(multi_class="auto", solver="lbfgs"), # fast ~10m and very good ~0.865 
#            QuadraticDiscriminantAnalysis(), # fast ~5m and very good ~0.81 
#            DecisionTreeClassifier(), # fast enough
#            RandomForestClassifier(), # fast enough
#            GaussianNB(), #very fast, performance around 0.58
         ]

non_models = [
            #xgb.XGBClassifier(objective="multi:softprob"), # too slow probably around 10h and okayish ~ 0.759
            KNeighborsClassifier(100), #works but v slow --> takes an hour + need to consider a hundred NNs because there are 20 classes
            SVC(kernel="linear", C=0.025, probability=True), # too slow to work
            SVC(gamma=2, C=1, probability=True), # too slow to work but v good ~0.83
            GaussianProcessClassifier(1.0 * RBF(1.0)), # too slow to work
            MLPClassifier(alpha=1, max_iter=1000), #fast enough ~80m and really good ~0.81
            AdaBoostClassifier() #prob take 1-2h, also very poor performance 
        ]

non_names = ["Nearest Neighbors", "Linear SVM","RBF SVM","Gaussian Process", "Neural Net","AdaBoost"] # "xgboost",

names = [
#         "Multinomial Naive Bayes",
         "Logistic Regression",
#         "QDA",
#         "Decision Tree",
#         "Random Forest",
#         "Gaussian Naive Bayes"
         ]

#"Nearest Neighbors", "Linear SVM" are ridiculously slow

def load_data(dataset='20newsgroups', true_ratio = 0.5):
    # true_ratio: float --> 0.01, 0.05, 0.1, 0.5 (but 0.5 as the default has already been done)
    num_1 = int(true_ratio * 10000) 
    num_0 = 10000 - num_1
    
    if dataset == '20newsgroups':
        categories = ['alt.atheism','sci.electronics']
        categories = ['alt.atheism',
             'comp.graphics',
             'comp.os.ms-windows.misc',
             'comp.sys.ibm.pc.hardware',
             'comp.sys.mac.hardware',
             'comp.windows.x',
             'misc.forsale',
             'rec.autos',
             'rec.motorcycles',
             'rec.sport.baseball',
             'rec.sport.hockey',
             'sci.crypt',
             'sci.electronics',
             'sci.med',
             'sci.space',
             'soc.religion.christian',
             'talk.politics.guns',
             'talk.politics.mideast',
             'talk.politics.misc',
             'talk.religion.misc']
        
        
        newsgroups = fetch_20newsgroups(categories=categories, subset='all', shuffle=True)  
        df = pd.DataFrame([newsgroups.data, newsgroups.target.tolist()]).T
        df.columns = ['text','target']
    elif dataset == 'sms':
        df = pd.read_csv('sms.csv')
        #df = pd.read_csv('errortextdetectionsystem/sougou.csv')
        # this has 5298 1s and 6264 0s
        df.columns = ['original_text','target']
        df['text'] = df['original_text'].apply(lambda x: ' '.join(w for w in jieba.cut(x)) if str(type(x)) == "<class 'str'>" else ' ')
    
    elif dataset == 'sougou':
        df = pd.read_csv('sougou.csv')
        df.columns = ['original_text','target']
        df['text'] = df['original_text'].apply(lambda x: ' '.join(w for w in jieba.cut(x)) if str(type(x)) == "<class 'str'>" else ' ')
        df_true = df.loc[df['target'] == 1]
        df_false = df.loc[df['target'] == 0]
        # this has 25000 1s and 25000 0s
        df_true = df_true.sample(num_1)
        df_false_equal = df_false.sample(num_0)
        df = df_true.append(df_false_equal, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

    elif dataset == 'jigsaw':
        df = pd.read_csv('jigsaw_hatespeech.csv')
        df['text'] = df['comment_text']
        df['target'] = (df['toxic'] + df['severe_toxic'] + df['obscene'] + df['threat'] + df['identity_hate']) > 0
        df['target'] = df['target'].astype(int)
        # 15924 1s and 140000 0s
        df_true = df.loc[df['target'] == 1]
        df_false = df.loc[df['target'] == 0]
        df_true = df_true.sample(num_1)
        df_false_equal = df_false.sample(num_0)
        df = df_true.append(df_false_equal, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
    
    elif dataset == 'quora':
        df = pd.read_csv('quora_insincere_questions_train.csv')
        df['text'] = df['question_text']
        # 80000 1s and 1.2m 0s
        df_true = df.loc[df['target'] == 1]
        df_false = df.loc[df['target'] == 0]
        df_true = df_true.sample(num_1)
        df_false_equal = df_false.sample(num_0)
        df = df_true.append(df_false_equal, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
    
    elif dataset == 'attitude':
        df = pd.read_csv('data_train_feat_clean.csv')
        df.append(pd.read_csv('data_test_feat_clean.csv'), ignore_index=True)
        df = df.reset_index(drop=True).sample(frac=1)
        y_plus_1 = df['target'].values + 1
        y_reverse = y_plus_1 % 2
        X, y = df.drop(['target'], axis=1), y_reverse
        # 0: 9177, 1: 859 (reverse)
        return X, y
    
    X = df[['text']]
    y = df[['target']]
    
    def convert_to_list(input_df):
        #input df is a pandas series
        return [i[0] for i in input_df.values.tolist()]
    
    X = convert_to_list(X)
    y = convert_to_list(y)
    #y = [i if i == 0 else 1 for i in y]
    y = np.asarray(y)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    clf = TruncatedSVD(300)
    X = clf.fit_transform(X)
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X,y

def active_learning_experiment(test_ratio = 0.8, initial_label_rate = 0.5, strategy="QueryInstanceQBC", model=MultinomialNB(alpha=.01)):
    
    al = AlExperiment(X, y, model=model, stopping_criteria='num_of_queries', stopping_value=20,)
    al.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_rate)
    #query strategies: QueryInstanceQBC, QueryInstanceUncertainty, QueryInstanceRandom, QureyExpectedErrorReduction, QueryInstanceQUIRE, QueryInstanceGraphDensity, QueryInstanceBMDR, QueryInstanceSPAL, QueryInstanceLAL, QueryExpectedErrorReduction
    #default: "QueryInstanceUncertainty"
    #if strategy == "QueryInstanceUncertainty":
    if '-' in strategy:
        strategy_split = strategy.split('-')
        if strategy_split[0] == "QueryInstanceQBC":
            al.set_query_strategy(strategy=strategy_split[0], disagreement=strategy_split[1])
        elif strategy_split[0] == "QueryInstanceUncertainty":
            al.set_query_strategy(strategy=strategy_split[0], measure=strategy_split[1])
    else:
        al.set_query_strategy(strategy=strategy) # measure= 'least_confident'--> default, 'entropy','margin' , 
    #al.set_performance_metric('accuracy_score')
    al.set_performance_metric('f1_score')
    al.start_query(multi_thread=False)
    #al.plot_learning_curve()
    results = [float(str(i).split('|')[-2].strip().split(' ')[0]) for i in al.get_experiment_result()]
    average = sum(results)/len(results)
    
    return average

def active_learning_experiment_mp(test_value_and_initial_label_rate):
    #test_value_and_initial_label_rate is a list with 2 floats
    #test_value_and_initial_label_rate[0] is test_value and test_value_and_initial_label_rate[1] is initial label rate
    test_ratio = test_value_and_initial_label_rate[0]
    initial_label_rate = test_value_and_initial_label_rate[1]
    #try:
    results = active_learning_experiment(test_ratio = test_ratio, initial_label_rate = initial_label_rate, strategy=strategy, model=model)
    shared_dict[tuple(test_value_and_initial_label_rate)] = results
    return test_value_and_initial_label_rate, results
    """
    except LinAlgError:
        pass
        return None
    """
    
def same_value_as_input(some_list):
    shared_dict[tuple(some_list)] = some_list[0] * some_list[1]
    return some_list

#if __name__ == '__main__':

#parser = argparse.ArgumentParser(description="preprocess data.csv to train_data and test_data, weibo.csv to data_weibo.csv ...")
#parser.add_argument("-d", "--dataset", default='20newsgroups', help="20newsgroups sms sougou quora jigsaw")
#args = parser.parse_args()
#dataset = args.dataset

#X,y = load_data(dataset=dataset)
    
true_ratio_formating = {
            0.05:"0p05",
            0.1:"0p1",
            0.5:"0p5"
        }

warnings.filterwarnings("ignore", category=Warning)

#X, y = load_data(dataset='attitude')

#dataset = 'sougou'
#true_ratio = 0.5 # the current issue is that at low ratios, the number of true sample is low and can be zero some times so need to redefine the sampler

datasets = ['attitude']
true_ratios = [0.5]

for dataset in datasets:
    for true_ratio in true_ratios:
        true_ratio_f = true_ratio_formating[true_ratio]
        
        X,y = load_data(dataset=dataset, true_ratio=true_ratio)
        
        #active_learning_experiment(test_ratio = 0.8, initial_label_rate = 0.5)
        learning_strategies = ["QueryInstanceUncertainty-least_confident",'QueryInstanceQBC-KL_divergence','QueryInstanceQBC-vote_entropy',"QueryInstanceUncertainty-margin","QueryInstanceUncertainty-entropy","QueryInstanceRandom"] #QBC takes around 20m #"QueryInstanceRandom", "QueryInstanceGraphDensity" --> takes a long time 3 hrs lol, "QueryInstanceBMDR"--> onlyfor binary, "QueryInstanceSPAL"--> only for binary, "QueryInstanceLAL", --> only for binary, "QueryExpectedErrorReduction" --> too slow, "QueryInstanceQUIRE" --> too slow 
        
        #learning_strategies = ["QueryInstanceUncertainty-margin"]
        #measures_QueryInstanceUncertainty = ['least_confident','entropy','margin'] # only applicable when "QueryInstanceUncertainty"
        #disagreement = ['KL_divergence','vote_entropy']
        range_of_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #
        #initial_label_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        initial_label_rates = [0.1, 0.5, 0.9] #0.1, 0.5, 
        #this takes 5 minutes --> dont use multi threading because it increases runtime and worsens performance
        
        list_of_combinations_test_value_and_initial_label_rate = []
        for i in range_of_values:
            for j in initial_label_rates:
                list1 = [i]
                list1.append(j)
                list_of_combinations_test_value_and_initial_label_rate.append(list1)
        
        
        for model_num in range(len(models)):
            model = models[model_num]
            model_name = names[model_num]
        #    model = MultinomialNB(alpha=.01)
        #    model_name = "Multinomial Naive Bayes"
            for strategy in learning_strategies:
                start_time = time.time()
                #single processing version
                """
                results_by_test_size = {}
                
                for i in range_of_values:
                    results_by_test_size[i] = []
                    for j in initial_label_rates:
                        results_by_test_size[i].append(active_learning_experiment(test_ratio = i, initial_label_rate=j, strategy=strategy, model=model))  
                        
                    print(str(i), ' done')
                """
                #multiprocessing version
                
                manager = multiprocessing.Manager()
                shared_dict = manager.dict()
                number_of_processes = 5
                p = multiprocessing.Pool(number_of_processes)
                xs = p.map(active_learning_experiment_mp, list_of_combinations_test_value_and_initial_label_rate)
                shared_dict = dict(shared_dict)
                results_by_test_size = {}
                for i in shared_dict:
                    if i[0] not in results_by_test_size:
                        results_by_test_size[i[0]] = {}
                    results_by_test_size[i[0]][i[1]] = shared_dict[i]
                
                
                filename = "%s_lsi_activelearning_label_rate_0p159_strategy_%s_model_%s_true_ratio_%s.json" % (dataset, strategy, model_name, true_ratio_f)
                with open(filename, "w") as write_file:
                    json.dump(results_by_test_size, write_file)
                print(time.time() - start_time)


          
   


#single process 1000s
#multiprocess 500s on mac
    
    
    
# for some reason the performance measured by al is a lot lower than scikit learn with the smae model and training data

# variables to play with --> initial label rate, test_ratio, query_strategy, query_measure and model