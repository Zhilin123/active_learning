#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:35:01 2019

@author: wangxiujiang
"""

import json
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
all_jsons = os.listdir("results")
active_filenames = [i for i in all_jsons if "activelearning" in i]
random_filenames = [i for i in all_jsons if "randomsampling" in i]

filename = all_jsons[25]
filename = "results/" + filename

def parse_active_filename(filename):
    dataset = filename.split("_lsi")[0]
    model_name = filename.split("model_")[1].split(".")[0]
    strategy = filename.split("strategy_")[1].split("_model_")[0]
    return [dataset, model_name, strategy]

def parse_random_filename(filename):
    dataset = filename.split("_lsi")[0]
    model_name = filename.split("model_")[1].split(".")[0]
    return [dataset, model_name, "random-sampling"]

def get_data_json(filename):
    
    with open(filename, "r") as read_file:
        data = json.load(read_file)
    
    range_of_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    #label_rate_0p1 =[]
    #label_rate_0p5 =[]
    label_rate_0p9 =[]
    
    for i in range_of_values:
        if type(data[str(i)]) == type([1]):
    #        label_rate_0p1.append(data[str(i)]["0.1"])
    #        label_rate_0p5.append(data[str(i)]["0.5"])
             label_rate_0p9.append(data[str(i)][-1])
        elif type(data[str(i)]) == type(1.1):
            label_rate_0p9.append(data[str(i)])
        else:
    #        label_rate_0p1.append(data[str(i)][0])
    #        label_rate_0p5.append(data[str(i)][1])
             label_rate_0p9.append(data[str(i)]["0.9"])
    
    range_of_train_size = [1 - i for i in range_of_values]
    assert len(label_rate_0p9) == 9
    return label_rate_0p9

all_data = {}

# note filename is without "results/"

for filename in active_filenames:
    one_data = parse_active_filename(filename)
    dataset_and_model = (one_data[0],one_data[1])
    strategy = one_data[2]
    if dataset_and_model not in all_data:
        all_data[dataset_and_model] = {}
    all_data[dataset_and_model][strategy] = get_data_json("results/"+filename)

for filename in random_filenames:
    one_data = parse_random_filename(filename)
    dataset_and_model = (one_data[0],one_data[1])
    strategy = one_data[2]
    if dataset_and_model not in all_data:
        all_data[dataset_and_model] = {}
    all_data[dataset_and_model][strategy] = get_data_json("results/"+filename)
    

dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]


for dataset_and_model in all_data:
    range_of_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    range_of_train_size = [1 - i for i in range_of_values]
    one_data = all_data[dataset_and_model]
    one_data['Proportion of training set'] = np.asarray(range_of_train_size)
    df = pd.DataFrame(one_data).set_index('Proportion of training set')
    title = dataset_and_model[0] + " " + dataset_and_model[1]
    sns.lineplot(data=df,dashes=dash_styles).set_title(title)
    plt.savefig(title+'.jpg')
    plt.clf()
        

"""
df = pd.DataFrame(
    {'Proportion of training set': np.asarray(range_of_train_size),
     'Initial Label Rate 0.1': np.asarray(label_rate_0p1),
     'Initial Label Rate 0.5': np.asarray(label_rate_0p5),
     'Initial Label Rate 0.9': np.asarray(label_rate_0p9),
    }).set_index('Proportion of training set')

sns.lineplot(data=df).set_title(filename)
"""
    
"""
with open("20newsgroup_lsi_randomsampling.json", "r") as read_file:
    random_sampling = json.load(read_file)

with open("20newsgroup_lsi_activelearning_label_rate_varying_least_confident.json", "r") as read_file:
    active_learning_least_confident = json.load(read_file)

with open("20newsgroup_lsi_activelearning_label_rate_varying_measure_margin.json", "r") as read_file:
    active_learning_margin = json.load(read_file)
    
with open("20newsgroup_lsi_activelearning_label_rate_0p9_measure_entropy.json", "r") as read_file:
    active_learning_entropy = json.load(read_file)

random_sampling_list = []

range_of_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range_of_values:
    random_sampling_list.append(random_sampling[str(i)])
    
active_learning_least_confident_list = []

for i in range_of_values:
    active_learning_least_confident_list.append(active_learning_least_confident[str(i)][-1])

active_learning_margin_list = []

for i in range_of_values:
    active_learning_margin_list.append(active_learning_margin[str(i)][-1])

active_learning_entropy_list = []

for i in range_of_values:
    active_learning_entropy_list.append(active_learning_entropy[str(i)][-1])

range_of_train_size = [1 - i for i in range_of_values]

df = pd.DataFrame(
    {'Proportion of training set': np.asarray(range_of_train_size),
     'Random Sampling': np.asarray(random_sampling_list),
     'Active Learning with margin as measure': np.asarray(active_learning_margin_list),
     'Active Learning with entropy as measure': np.asarray(active_learning_entropy_list),
     'Active Learning with least confident as measure': np.asarray(active_learning_least_confident_list)
    }).set_index('Proportion of training set')

sns.lineplot(data=df).set_title(" 20 News Group task / MultinomialNB / Active learning with initial label rate = 0.9")

"""

