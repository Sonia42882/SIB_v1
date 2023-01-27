'''
A alterar:

    - Poucos descritores: gerar mais com o propythia
    - Separar o x do y antes de fazer standardscaler e usar o y sem alterar os valores 
    - Retirar valores de pH maiores que 14
    - Fazer análise com base na literatura, exemplo, ponto em que a termoestabilidade representa a estabilidade etc
    - Aumentar número de features, 
    - não e preciso normalizar (podemos fazer scale e apenas no X), 
    - machine learning com regressão (tm variável continua)
'''


import pandas as pd
import scipy.cluster.hierarchy
import seaborn as sn
import numpy as np
from tensorflow import keras
from keras.optimizers.schedules import *
from propythia import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import r_regression
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind
from statsmodels.stats.weightstats import ztest as ztest
import statsmodels.api as statsmodels
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
# %matplotlib inline

import pandas as pd
import numpy as np
import re
import random
import requests as r
from Bio import SeqIO
from io import StringIO
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.optimizers.schedules import *

from propythia import *
from propythia.protein_descriptores import ProteinDescritors

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import auc, roc_curve, matthews_corrcoef, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, hinge_loss, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, SelectPercentile

import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', delimiter = ',')
updates = pd.read_csv('train_updates_20220929.csv', delimiter = ',')

print(train.shape,train.head(),updates.shape,updates.head())


#colunas para fazer o update
col_to_update = ['protein_sequence', 'pH', 'tm']

#seq_id update
to_update_id = list(updates[~updates.protein_sequence.isna()].seq_id)

#faz o update só nas linhas que são necessarias
updates = updates[updates.seq_id.isin(to_update_id)].set_index('seq_id')

#update dos valores no dataset de treino
train.loc[to_update_id, col_to_update] = updates[col_to_update].values

#verificar se está com os updates
pd.concat([train[train.seq_id.isin(to_update_id)].set_index('seq_id'), updates], axis = 1)

#nao precisamos dos links - drop do data_source
train.drop(['data_source'], axis=1, inplace=True)

print(train.shape,train.head())



#obter os descritores
descriptors_df = ProteinDescritors(dataset= train ,  col= 'protein_sequence')

def_len = descriptors_df.get_lenght(n_jobs=4)

def_aa = descriptors_df.get_aa_comp(n_jobs=4)

df_all_phycoch_descriptors = descriptors_df.get_all_physicochemical(ph=7, amide=False, n_jobs=4)

all_descriptors = pd.merge(descriptors_df,def_len,def_aa,df_all_phycoch_descriptors)

all_descriptors.to_csv('descriptors_novo.csv')
all_descriptors


############# como faço para dar?????? pq é q tudo é importado direitinho no noteb e aqui não???????????????????????? fazer instalações diretamente no /usr/bin/python3??????????????????????? python 3.8.10