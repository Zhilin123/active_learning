#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:55:33 2019

@author: wangxiujiang
"""


# Task: to test the effectiveness of active learning wehn applied to various algorithms for text classification tasks similar to classifying teachers' attitudes

# also try to learn pytorch, tensorflow, scikit-learn, data visualisation and maybe machine learning explanability in this project --> write this in LaTeX and in the format of a common paper


# 1 Obtain raw text --> the final product should be in chinese but can do some experiments in English forums first such as Reddit 

#maybe start with the 20 newsgroup and use somethng like politics vs no politics 20 news groups; sms.csv, sougou.csv or https://www.kaggle.com/mswarbrickjones/reddit-selfposts --> reduce it to a 2 class problem or https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview or like https://www.kaggle.com/c/quora-insincere-questions-classification/data

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import json
import jieba
import os
import sklearn
import warnings

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
import xgboost as xgb

warnings.filterwarnings("ignore", category=Warning)


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
        
    return df

def train_data(test_size=0.8, model=MultinomialNB(alpha=.01), df=None, is_attitude=False):
    
    if is_attitude:
        y_plus_1 = df['target'].values + 1
        y_reverse = y_plus_1 % 2
        X, y = df.drop(['target'], axis=1), y_reverse
        # 0: 9177, 1: 859 (reverse)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)
        clf = model
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        return sklearn.metrics.f1_score(y_test, pred)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(df[['text']],df[['target']], test_size = test_size)
    
        def convert_to_list(input_df):
            #input df is a pandas series
            return [i[0] for i in input_df.values.tolist()]
        
        
        X_train = convert_to_list(X_train)
        X_test = convert_to_list(X_test)
        y_train = convert_to_list(y_train)
        y_test = convert_to_list(y_test)
        # change the task to whether it is 'alt.atheism'
        #y_train = [i if i == 0 else 1 for i in y_train]
        #y_test = [i if i == 0 else 1 for i in y_test]
        
       
        
        
        #X_train, X_test, y_train, y_test = load_data()
        
        # 2 Preprocessing --> this depends on whether the model used BoW
        
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train+X_test)
        vectors_train = vectorizer.transform(X_train)
        vectors_test = vectorizer.transform(X_test)
        
        clf = TruncatedSVD(300)
        vectors_train = clf.fit_transform(vectors_train)
        vectors_test = clf.transform(vectors_test)
        scaler = MinMaxScaler()
        scaler.fit(vectors_train)
        vectors_train = scaler.transform(vectors_train)
        vectors_test = scaler.transform(vectors_test)
        
        # 3 Neural Networks
        
        clf = model
        clf.fit(vectors_train, y_train)
        pred = clf.predict(vectors_test)
        #print(metrics.classification_report(y_test, pred))
        #return test_size, metrics.precision_recall_fscore_support(y_test, pred)
        #return metrics.accuracy_score(y_test, pred)
        return sklearn.metrics.f1_score(y_test, pred)

#df = load_data()
#train_data(test_size=0.1, model=SVC(gamma=2, C=1))

#all_results = {}
#
#i = 0.05
#while i < 0.99:
#    results = train_data(train_size=i)
#    all_results[results[0]] = results[1]
#    i += 0.05

all_jsons = os.listdir("results")
active_filenames = [i for i in all_jsons if "activelearning" in i and "_true_ratio_" in i and "attitude" in i]
#random_filenames = [i for i in all_jsons if "randomsampling" in i]

done = []
"""
for random_filename in random_filenames:
    dataset = random_filename.split("_lsi")[0]
    model_name = random_filename.split("model_")[1].split(".")[0]
    done.append([dataset,model_name])
"""

true_ratio_formating = {
            0.05:"0p05",
            0.1:"0p1",
            0.5:"0p5"
        }
reverse_true_ratio = {
            "0p05":0.05,
            "0p1":0.1,
            "0p5":0.5
        }
#active_filename = "20newsgroups_lsi_activelearning_label_rate_0p159_strategy_QueryInstanceUncertainty-least_confident_model_Logistic Regression.json"
for active_filename in active_filenames:
    dataset = active_filename.split("_lsi")[0]
    model_name = active_filename.split("model_")[1].split("_true_ratio_")[0]
    split_ratio = active_filename.split("_true_ratio_")[1].split('.')[0]

    
    if [dataset, model_name,split_ratio] not in done:
        df = load_data(dataset=dataset, true_ratio=reverse_true_ratio[split_ratio])
        model = None
        for i in range(len(names)):
            if names[i] == model_name:
                model = models[i]
        results_by_test_size = {}
        range_of_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        print(dataset, model_name)
        for i in range_of_values:
            all_results = []
            for time in range(3):
                all_results.append(train_data(test_size=i, model=model,df=df,is_attitude=True))
            results_by_test_size[i] = sum(all_results) / len(all_results)
            print(i, " done")
        
        random_filename = "%s_lsi_randomsampling_model_%s_true_ratio_%s.json" % (dataset, model_name,split_ratio)
        random_filename = "results/" + random_filename
        
        with open(random_filename, "w") as write_file:
            json.dump(results_by_test_size, write_file)
        
        done.append([dataset,model_name,split_ratio])
    
# for random modelling might need to do it ten times and take average
"""





"""

    

"""
x_axis = []
precision_class0 = []
recall_class0 =[]
fscore_class0 = []


for i in all_results:
    x_axis.append(i)
    precision_class0.append(all_results[i][0][0])
    recall_class0.append(all_results[i][1][0])
    fscore_class0.append(all_results[i][2][0])

percentile_list = pd.DataFrame(
    {'Proportion of train data': x_axis,
     'precision_class0': precision_class0,
     'recall_class0': recall_class0,
     'fscore_class0': fscore_class0
    }).set_index('Proportion of train data')

sns.lineplot(data=percentile_list).set_title("Policy: Random Sampling/ 20 News Group / MultinomialNB")
"""

"""
# try to use some of them in pytorch or tensorflow
Sklearn --> classification model (maybe even something like naive bayes) including logistic regression
LSI --> I probably have the code for this alr
Dense Neural Network
CNN
LSTM
BERT
TransformerXL
XLNET
FrameNet-based
"""

#4 do evaluation


#5 try various built in strategies in active learning - and see where it works and whether it doesn't

#6 write a report and report the level of improvement in a standardised manner
