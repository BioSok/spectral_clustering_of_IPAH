
# %% 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import time
import os
import csv
import xlrd
import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import shap
import xgboost
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.feature_selection import RFECV

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# memory management
import gc

# utilities
from itertools import chain
set_matplotlib_formats('png', 'pdf') # uses vector figures in pdf exports --
plt.style.use('seaborn-pastel')

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, silhouette_score
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Supress Warnings
warnings.filterwarnings("ignore")

# %%

''' Create folder '''
def mkdir(dirName):

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ",end = '')
    else:    
        print("Directory " , dirName ,  " already exists",end = '')

''' Load dataset  '''
def load_dataset(fName):
    df=pd.read_csv(fName).dropna()

    return df

''' Load dataset  '''
def load_dataset2(fName, features):
    df=pd.read_csv(fName).dropna()
    df=df.drop(columns=features)
    
    return df

''' Extract Cluster A dataset  '''
def extract_cluster0_data(fName):
    df=load_dataset(fName)

    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    cols=df.select_dtypes(include=['boolean']).columns # identify categorical variables
    df[cols] = df[cols].astype(int) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df0=df

    ''' Create a dataset of cluster 0 vs rest '''
    df0['cluster6']=df0.cluster6.replace({0:10});
    df0['cluster6']=df0.cluster6.replace({1:0, 2:0, 3:0, 4:0, 5:0, 10:1});

    # create data and labels
    df0_data=df0.drop(columns=['cluster6'])
    df0_labels=df0['cluster6']

    #return df0, df0_data, df0_labels
    return df0

''' Extract Cluster B dataset  '''
def extract_cluster1_data(fName):
    df=load_dataset(fName)

    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    cols=df.select_dtypes(include=['boolean']).columns # identify categorical variables
    df[cols] = df[cols].astype(int) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df1=df

    ''' Create a dataset of cluster 0 vs rest '''
    #df1['cluster6']=df1.cluster6.replace({0:10});
    df1['cluster6']=df1.cluster6.replace({0:0, 1:1, 2:0, 3:0, 4:0, 5:0});

    # create data and labels
    df1_data=df1.drop(columns=['cluster6'])
    df1_labels=df1['cluster6']

    #return df0, df0_data, df0_labels
    return df1

''' Extract Cluster C dataset  '''
def extract_cluster2_data(fName):
    df=load_dataset(fName)

    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    cols=df.select_dtypes(include=['boolean']).columns # identify categorical variables
    df[cols] = df[cols].astype(int) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df2=df

    ''' Create a dataset of cluster 0 vs rest '''
    #df1['cluster6']=df1.cluster6.replace({0:10});
    df2['cluster6']=df2.cluster6.replace({0:0, 1:0, 2:1, 3:0, 4:0, 5:0});

    # create data and labels
    df2_data=df2.drop(columns=['cluster6'])
    df2_labels=df2['cluster6']

    #return df0, df0_data, df0_labels
    return df2

''' Extract Cluster D dataset  '''
def extract_cluster3_data(fName):
    df=load_dataset(fName)

    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    cols=df.select_dtypes(include=['boolean']).columns # identify categorical variables
    df[cols] = df[cols].astype(int) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df3=df

    ''' Create a dataset of cluster 0 vs rest '''
    #df1['cluster6']=df1.cluster6.replace({0:10});
    df3['cluster6']=df3.cluster6.replace({0:0, 1:0, 2:0, 3:1, 4:0, 5:0});

    # create data and labels
    df3_data=df3.drop(columns=['cluster6'])
    df3_labels=df3['cluster6']

    #return df0, df0_data, df0_labels
    return df3


''' Extract Cluster E dataset  '''
def extract_cluster4_data(fName):
    df=load_dataset(fName)

    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    cols=df.select_dtypes(include=['boolean']).columns # identify categorical variables
    df[cols] = df[cols].astype(int) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df4=df

    ''' Create a dataset of cluster 0 vs rest '''
    #df1['cluster6']=df1.cluster6.replace({0:10});
    df4['cluster6']=df4.cluster6.replace({0:0, 1:0, 2:0, 3:0, 4:1, 5:0});

    # create data and labels
    df4_data=df4.drop(columns=['cluster6'])
    df4_labels=df4['cluster6']

    #return df0, df0_data, df0_labels
    return df4

''' Extract Cluster F dataset  '''
def extract_cluster5_data(fName):
    df=load_dataset(fName)

    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    cols=df.select_dtypes(include=['boolean']).columns # identify categorical variables
    df[cols] = df[cols].astype(int) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df5=df

    ''' Create a dataset of cluster 0 vs rest '''
    #df1['cluster6']=df1.cluster6.replace({0:10});
    df5['cluster6']=df5.cluster6.replace({0:0, 1:0, 2:0, 3:0, 4:0, 5:1});

    # create data and labels
    df5_data=df5.drop(columns=['cluster6'])
    df5_labels=df5['cluster6']

    #return df0, df0_data, df0_labels
    return df5

def extract_cluster0_data2(fName,features):
    #mkdir('Analysis')
    #mkdir('Analysis/data')
    
    #fName='../data/Imputedset1.csv'
    #df=load_dataset(fName).drop(columns=['site','SampleID'])
    df=load_dataset(fName).drop(columns=features)

    
    # LABEL ENCODING ---
    cols=df.select_dtypes(include=['object']).columns # identify categorical variables
    df[cols] = df[cols].apply(LabelEncoder().fit_transform) # Encoding
    
    ''' Create dataset for cluster 0 '''
    df0=df

    ''' Create a dataset of cluster 0 vs rest '''
    df0['cluster6']=df0.cluster6.replace({0:10});
    df0['cluster6']=df0.cluster6.replace({1:0, 2:0, 3:0, 4:0, 5:0, 10:1});

    # create data and labels
    df0_data, df0_labels=df0.drop(columns=['cluster6']),df0['cluster6']

    ''' save data '''
    #exec("df0.to_csv('Analysis/data/%s"%fName+".csv',index=False)")

    ''' Plot membership '''
    #sns.set(style="ticks")
    #sns.countplot(x="cluster6", data=df0, palette="bwr")
    #plt.show()
    #plt.title('Cluster 0 membership')
    #plt.ylabel('Count')
    #plt.xlabel('Membership')
    
    return df0, df0_data, df0_labels


'''
    Use boruta to rankfeatures and select optimal features
    
    Return: 
    ranking: ranking of all features
    selected: selected optimal features
'''
def boruta_feature_ranking(data, labels):
    import numpy as np 
    import pandas as pd 
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=4242, max_iter = 100, perc = 90)
    boruta_feature_selector.fit(data.values, labels.values)

    X_filtered = boruta_feature_selector.transform(data.values)
    X_filtered.shape

    final_features = list()
    indexes = np.where(boruta_feature_selector.support_ == True)

    features=data.columns
    for x in np.nditer(indexes):
        final_features.append(features[x])
    #print(final_features)

    raw_ranking=pd.DataFrame({'feature':data.columns,'boruta_rank':boruta_feature_selector.ranking_})
    ranking=pd.DataFrame({'feature':data.columns,'boruta_rank':boruta_feature_selector.ranking_}).sort_values(by='boruta_rank', ascending=True)
    
    selected=pd.DataFrame({'feature':final_features})
    #print(selected)
    
    #boruta_ranking['rank'].sum()
    ranking['boruta_nr']=ranking['boruta_rank']/ranking['boruta_rank'].sum().round(3)
    return ranking, raw_ranking, selected


#Function to test model accuracy based on testing data
def accuracy(model, x_test, y_test):
    
    prediction = model.predict(x_test)
    acc = accuracy_score(y_test, prediction) * 100
    
    return acc

def shap_feature_ranking(data,target):
    import pandas as pd
    import numpy as np
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    # Classifier
    model = GradientBoostingClassifier(n_estimators=1000, max_depth=10, learning_rate=0.001)

    # Fit the Model
    model.fit(data, target)

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    shap_ranking=pd.DataFrame(pd.DataFrame(shap_values, columns=data.columns).sum(),columns=['importance'])
    shap_ranking['absImp']=abs(shap_ranking.importance)
    shap_ranking=shap_ranking.sort_values(by='absImp', ascending=False).drop(columns=['absImp'])
    shap_ranking=shap_ranking.reset_index()
    shap_ranking['shap_nr']=abs(shap_ranking['importance'])/abs(shap_ranking.importance).sum().round(3)
    
    unranked_shap_ranking=pd.DataFrame(pd.DataFrame(shap_values, columns=data.columns).sum(),columns=['importance'])
    unranked_shap_ranking['shap_nr']=abs(unranked_shap_ranking['importance'])/abs(unranked_shap_ranking.importance).sum().round(3)
    
    return shap_ranking, unranked_shap_ranking

def rfecv_ranking(data,target, num_to_select):
    from sklearn.linear_model import LogisticRegression
    estimator = LogisticRegression(random_state=220)
    rfecv = RFECV(estimator=estimator, cv=StratifiedKFold(10, random_state=220, shuffle=True), min_features_to_select=1, scoring="accuracy")
    rfecv.fit(data, target)
    rfecv_feature_ranking=pd.DataFrame({'feature':data.columns,'rfecv_rank':rfecv.ranking_}).sort_values(by='rfecv_rank', ascending=True)
    unranked_rfecv_ranking = pd.DataFrame({'feature':data.columns,'rfecv_rank':rfecv.ranking_})

    return rfecv_feature_ranking, unranked_rfecv_ranking

def rfe_with_cv(data,target, num_to_select):
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy', min_features_to_select=num_to_select)
    rfecv.fit(X, y)
    
    # optimal number of features 
    n_features=rfecv.n_features_
    grid_scores_=rfecv.grid_scores_

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    
    return scores

def rank_features(data, target):
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    import xgboost as xgb
    
    # dt
    estimator=DecisionTreeClassifier()
    rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    dt_ranking=rfe.ranking_
    features= data.columns
    #print(' xgboost ',end='')
    
    # xgb
    estimator=xgb.XGBClassifier(verbosity =0)
    rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    xgb_ranking=rfe.ranking_
    features= data.columns
    #print(' xgb ',end='')
    
    # gb
    estimator=GradientBoostingClassifier()
    rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    gb_ranking=rfe.ranking_
    features= data.columns
    #print(' gb ',end='')
    
    # rf
    estimator=RandomForestClassifier()
    rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    rf_ranking=rfe.ranking_
    features= data.columns
    #print(' rf ',end='')
    
    # lr
    #estimator=LogisticRegression()
    #rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    #lr_ranking=rfe.ranking_
    #features= data.columns
    #print(' lr ',end='')
    
    # svc
    #estimator=SVC(kernel='linear',gamma='auto')
    #rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    #svc_ranking=rfe.ranking_
    #features= data.columns
    #print(svc)
    
    # dt
    estimator=DecisionTreeClassifier()
    rfe = RFE(estimator, n_features_to_select=1).fit(data, target)
    dt_ranking=rfe.ranking_
    features= data.columns

    rankings =pd.DataFrame({'feature':features, 'dt':dt_ranking, 'xgb':xgb_ranking, 'gb':gb_ranking, 'rf':rf_ranking})
    
    rankings['total']=rankings.sum(axis=1)
    rankings['normalised_rank']=rankings.total/np.sum(rankings.total)
    
    return rankings
    
def evaluate_signatures(X,y):
    
    # explore the algorithm wrapped by RFE
    from numpy import mean
    from numpy import std
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Perceptron
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from matplotlib import pyplot
    
    # get a list of models to evaluate
    def get_models():
            models = dict()
            # lr
            models['lr'] = LogisticRegression()

            # perceptron
            models['per'] = Perceptron()

            # cart
            models['dt'] = DecisionTreeClassifier()

            # rf
            models['rf'] = RandomForestClassifier()

            # gbm
            models['gbm'] = GradientBoostingClassifier()
            
            #svc
            models['svc'] = SVC(kernel='rbf',gamma='auto')

            return models
        
    # evaluate a give model using cross-validation
    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores
    
    # get the models to evaluate
    models = get_models()
    
    # evaluate the models and store results
    results, names = list(), list()
    
    # evaluate the models
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()
    
    # save the results
    perf=pd.DataFrame({'model':names, 'perf':results})
    
    return perf

''' Feature selection '''
#def select_signature(X,y,target):
def select_signature(X,y,size):
   
    ''' split train and test sets  '''
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=220)
    #X_train.shape, X_test.shape
    
    '''  Remove constant and quasi-constant features  '''
    quasi_constant_feat = []

    # iterate over every feature
    for feature in X_train.columns:

        # find the predominant value, that is the value that is shared
        # by most observations
        predominant = (X_train[feature].value_counts() / np.float(
            len(X_train))).sort_values(ascending=False).values[0]

        # evaluate the predominant feature: do more than 99% of the observations
        # show 1 value?
        if predominant > 0.998:

            # if yes, add the variable to the list
            quasi_constant_feat.append(feature)

    X_train.drop(labels=quasi_constant_feat, axis=1, inplace=True)
    X_test.drop(labels=quasi_constant_feat, axis=1, inplace=True)

    #print('Shape: ',X_train.shape, X_test.shape)
    
    
    ''' Remove duplicated features '''
    duplicated_feat = []
    for i in range(0, len(X_train.columns)):
        if i % 10 == 0:  # this helps me understand how the loop is going
            pass#print(i)

        col_1 = X_train.columns[i]

        for col_2 in X_train.columns[i + 1:]:
            if X_train[col_1].equals(X_train[col_2]):
                duplicated_feat.append(col_2)

    #len(duplicated_feat)


    # remove duplicated features
    X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
    X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

    #print('shape: ',X_train.shape, X_test.shape)
   
    # check classification perf of all features
    xgb_model = XGBClassifier(verbosity = 0).fit(X_train, y_train)

    # predict
    xgb_y_predict = xgb_model.predict(X_test)

    # accuracy score
    xgb_score = accuracy_score(xgb_y_predict, y_test)

    #print('Accuracy score is:', xgb_score)
    
    
    '''
    Synthetic Minority Oversampling Technique (SMOTE)
    One way to fight imbalance data is to generate new samples in the minority classes. The most naive strategy generate new samples by randomly sampling with replacement currently available samples.

    1) Choose a minority class as the input vector

    2) Find its k nearest neighbors (k_neighbors is specified as an argument in the SMOTE() function)

    3) Choose one of these neighbors and place a synthetic point anywhere on the line joining the point under consideration and its chosen neighbor

    4) Repeat the steps until data is balanced

    '''
    # import library
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()

    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(X, y)

    # use oversampled data
    X, y = x_smote, y_smote 


    #print('Used Oversampled shape: ', X.shape)
    
    '''   Feature selection  '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=220)

    #classifier = LogisticRegression(random_state=0, penalty='l2')

    # randomForestClassifier
    classifier  = RandomForestClassifier(max_depth=2, random_state=220)

    # SVC
    #classifier  = SVC(kernel='linear',gamma='auto')

    #Select best feature 
    #rfe = RFE(classifier, n_features_to_select= None)
    rfe = RFE(classifier, n_features_to_select= size)
    rfe = rfe.fit(X_train, y_train)

    #Summarize the selection of the attributes
    #print(rfe.support_)
    #print(rfe.ranking_)
    #X_train.columns[rfe.support_]

   
    # store selected signatures
    selected = X_train.columns[rfe.support_]
    ranking = rfe.ranking_

    ''' Check correlation of selected features'''
    # New Correlation Matrix
    sns.set(style="white")

    # Compute the correlation matrix
    corr = X_train[X_train.columns[rfe.support_]].corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(18, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}) 
    
    plt.close()
    
    # check classification perf signature
    xgb_model = XGBClassifier(verbosity = 0).fit(X_train, y_train)

    # predict
    xgb_y_predict = xgb_model.predict(X_test)

    # accuracy score
    xgb_score = accuracy_score(xgb_y_predict, y_test)
    
    #print('Signature accuracy: ', xgb_score)
    
    return selected, ranking, xgb_score


# Feature analysis and selection
# feature selector class
class FeatureSelector():
    """  
        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        5. Find low importance features that do not contribute to a specified cumulative feature importance from the gbm
        
    Parms
    --------
        data : dataframe
 
        labels : array or series, default = None
    """
    
    def __init__(self, data, labels=None):
        
        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')
        
        self.base_features = list(data.columns)
        self.one_hot_features = None
        
        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None
        
        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None
        
        # Dictionary to hold removal operations
        self.ops = {}
        
        self.one_hot_correlated = False
        
    def identify_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above `missing_threshold`"""
        
        self.missing_threshold = missing_threshold

        # Calculate the fraction of missing in each column 
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending = False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns = {'index': 'feature', 0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.ops['missing'] = to_drop
        
        #print('%d features with greater than %0.2f missing values.\n' % (len(self.ops['missing']), self.missing_threshold))
        
    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending = True)
        
        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})

        to_drop = list(record_single_unique['feature'])
    
        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop
        
        #print('%d features with a single unique value.\n' % len(self.ops['single_unique']))
    
    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features. 
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal. 

        Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
        
        Parameters
        --------

        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features

        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients

        """
        
        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
        
         # Calculate the correlations between every column
        if one_hot:
            
            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)
            
            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()
        
        self.corr_matrix = corr_matrix
    
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        
        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:

            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]    

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                             'corr_feature': corr_features,
                                             'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index = True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop
        
        #print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']), self.correlation_threshold))

    def identify_zero_importance(self, task, eval_metric=None, n_iterations=10, early_stopping = True):

        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")
            
        if self.labels is None:
            raise ValueError("No training labels provided.")
        
        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1, ))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))
        
        #print('Training Gradient Boosting Model\n')
        
        # Iterate through each fold
        for _ in range(n_iterations):

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)

            else:
                raise ValueError('Task must be either "classification" or "regression"')
                
            # If training using early stopping need a validation set
            if early_stopping:
                
                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15, stratify=labels)

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric = eval_metric,
                          eval_set = [(valid_features, valid_labels)],
                          early_stopping_rounds = 100, verbose = -1)
                
                # Clean up memory
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()
                
            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
        
        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop
        
        #print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))
    
    def identify_low_importance(self, cumulative_importance):
         self.cumulative_importance = cumulative_importance
        
        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined. 
                                         Call the `identify_zero_importance` method first.""")
            
        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop
        
    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")
        
        self.reset_plot()
        
        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize = (7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins = np.linspace(0, 1, 11), edgecolor = 'k', color = 'red', linewidth = 1.5)
        plt.xticks(np.linspace(0, 1, 11));
        plt.xlabel('Missing Fraction', size = 14); plt.ylabel('Count of Features', size = 14); 
        plt.title("Fraction of Missing Values Histogram", size = 16);
        
    
    def plot_unique(self):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')
        
        self.reset_plot()

        # Histogram of number of unique values
        self.unique_stats.plot.hist(edgecolor = 'k', figsize = (7, 5))
        plt.ylabel('Frequency', size = 14); plt.xlabel('Unique Values', size = 14); 
        plt.title('Number of Unique Values Histogram', size = 16);
        
    
    def plot_collinear(self, plot_all = False):
        
        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')
        
        if plot_all:
            corr_matrix_plot = self.corr_matrix
            title = 'All Correlations'
        
        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])), list(set(self.record_collinear['drop_feature']))]

            title = "Correlations Above Threshold"

        f, ax = plt.subplots(figsize=(10, 8))
        
        # Diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0, linewidths=.25, cbar_kws={"shrink": 0.6})

        # Set the ylabels 
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));

        # Set the xlabels 
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));
        plt.title(title, size = 14)
        
    def plot_feature_importances(self, plot_n = 15, threshold = None):
        
        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')
            
        # Need to adjust number of features if greater than the features in the data
        if plot_n > self.feature_importances.shape[0]:
            plot_n = self.feature_importances.shape[0] - 1

        self.reset_plot()
        
        # Make a horizontal bar chart of feature importances
        plt.figure(figsize = (10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.feature_importances.index[:plot_n]))), 
                self.feature_importances['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(self.feature_importances.index[:plot_n]))))
        ax.set_yticklabels(self.feature_importances['feature'][:plot_n], size = 12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size = 16); plt.title('Feature Importances', size = 18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize = (6, 4))
        plt.plot(list(range(1, len(self.feature_importances) + 1)), self.feature_importances['cumulative_importance'], 'r-')
        plt.xlabel('Number of Features', size = 14); plt.ylabel('Cumulative Importance', size = 14); 
        plt.title('Cumulative Feature Importance', size = 16);

        if threshold:

            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
            plt.vlines(x = importance_index + 1, ymin = 0, ymax = 1, linestyles='--', colors = 'blue')
            plt.show();

            #print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
