#!/usr/bin/env python
# coding: utf-8

import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# for the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer


# for feature engineering
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

# for the model
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

# for feature engineering
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# from feature-engine
#from feature_engine import missing_data_imputers as mdi
from feature_engine import missing_data_imputers as mdi
from feature_engine import discretisers as dsc
from feature_engine import categorical_encoders as ce
import feature_engine.missing_data_imputers as mdi
from feature_engine.categorical_encoders import RareLabelCategoricalEncoder

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


from pathlib import Path
models_folder = Path("/Users/joaosantos/Documents/Projects/Insurance_Churn/Insurance_Churn/models/")
data_folder = Path("/Users/joaosantos/Documents/Projects/Insurance_Churn/Insurance_Churn/data/")


# Review this :https://www.kaggle.com/udita3996/eda-woe-iv-calc-model-training


data = pd.read_csv(data_folder/'raw/Train.csv')

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()


len(data)


# Check if missing values
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
# source: https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe

missing_values_table(data)


# Find Continuous and Categorical Features
def featureType(df):
    import numpy as np 
    from pandas.api.types import is_numeric_dtype

    columns = df.columns
    rows= len(df)
    colTypeBase=[]
    colType=[]
    for col in columns:
        try:
            try:
                uniq=len(np.unique(df[col]))
            except:
                 uniq=len(df.groupby(col)[col].count())
            if rows>10:
                if is_numeric_dtype(df[col]):
                    
                    if uniq==1:
                        colType.append('Unary')
                        colTypeBase.append('Unary')
                    elif uniq==2:
                        colType.append('Binary')
                        colTypeBase.append('Binary')
                    elif rows/uniq>3 and uniq>5:
                        colType.append('Continuous')
                        colTypeBase.append('Continuous')
                    else:
                        colType.append('Continuous-Ordinal')
                        colTypeBase.append('Ordinal')
                else:
                    if uniq==1:
                        colType.append('Unary')
                        colTypeBase.append('Category-Unary')
                    elif uniq==2:
                        colType.append('Binary')
                        colTypeBase.append('Category-Binary')
                    else:
                        colType.append('Categorical-Nominal')
                        colTypeBase.append('Nominal')
            else:
                if is_numeric_dtype(df[col]):
                    colType.append('Numeric')
                    colTypeBase.append('Numeric')
                else:
                    colType.append('Non-numeric')
                    colTypeBase.append('Non-numeric')
        except:
            colType.append('Issue')
                
    # Create dataframe    
    df_out =pd.DataFrame({'Feature':columns,
                          'BaseFeatureType':colTypeBase,
                        'AnalysisFeatureType':colType})
    return df_out

featureType(data)  


#import pandas_profiling
#data.profile_report()


# # Understand Types

# let's inspect the type of variables in pandas
data.dtypes


# let's inspect the variable values
for var in data.columns:
    print(var, data[var].unique()[0:20], '\n')


# # Distribuition of Target Variable

(data.groupby('labels')['labels'].count() / len(data)).plot.bar()


# # Identify data Types

# make list of variables  types
dates = [var for var in data.columns if 'Date' in var or 'Year' in var]


# numerical: discrete vs continuous
discrete = [var for var in data.columns if data[var].dtype!='O' and var!='label' and 2< data[var].nunique()<20 and var not in dates]
continuous = [var for var in data.columns if data[var].dtype!='O' and var!='label' and var not in discrete and var not in dates]
binary_num = [var for var in data.columns if data[var].nunique()==2 and  data[var].dtype!='O' and var!='label']
binary_cat = [var for var in data.columns if data[var].nunique()==2 and  data[var].dtype=='O' and var!='label']
# mixed
mixed = []

# categorical
categorical = [var for var in data.columns if data[var].dtype=='O' and var!='label' and var not in mixed and var not in binary_cat and var not in binary_num]

print('There are {} date variables'.format(len(dates)))
print('There are {} discrete variables'.format(len(discrete)))
print('There are {} binary categorical variables'.format(len(binary_cat)))
print('There are {} binary numeric variables'.format(len(binary_num)))
print('There are {} continuous variables'.format(len(continuous)))
print('There are {} categorical variables'.format(len(categorical)))
print('There are {} mixed variables'.format(len(mixed)))


continuous


dates


discrete


binary_cat


binary_num


categorical


# # Missing Values analisys

# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>=1]

# print the variable name and the percentage of missing values
for var in vars_with_na:
    print(var, np.round(data[var].isnull().mean(), 3),  ' % missing values')
    


def analyse_na_value(df, var):
    df = df.copy()

    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)

    # let's calculate the mean SalePrice where the information is missing or present
    df.groupby(var)['label'].count().plot.bar()
    plt.title(var)
    plt.show()


for var in vars_with_na:
    analyse_na_value(data, var)
    


# # Cardinality (number of different categories)

# cardinality (number of different categories)

data[categorical+mixed+discrete].nunique()


# # Distribuition of the Variables - Levels

# ## Continuous

# let's make boxplots to visualise outliers in the continuous variables 
# and histograms to get an idea of the distribution

for var in continuous:
    plt.figure(figsize=(6,4))
    plt.subplot(1, 2, 1)
    fig = data.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = data[var].hist(bins=200)
    fig.set_ylabel('Number of Quotes')
    fig.set_xlabel(var)

    plt.show()


# ## Discrete

# outliers in discrete variables
for var in discrete+categorical:
    (data.groupby(var)[var].count() / np.float(len(data))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()


# # Binary data

# outliers in discrete variables
for var in binary_num+binary_cat:
    (data.groupby(var)[var].count() / np.float(len(data))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()


# ## Rare Labels

#def analyse_rare_labels(df, var, rare_perc):
#    df = df.copy()
#    tmp = df.groupby(var)['SalePrice'].count() / len(df)
#    return tmp[tmp<rare_perc]

#for var in categorical:
#    print(analyse_rare_labels(data, var, 0.01))
#    print()


#for var in categorical:
#    plt.figure(figsize=(6,4))
#    plt.subplot(1, 2, 1)
#    fig1 = data.groupby([var,'QuoteConversion_Flag']).size().unstack().plot(kind='bar', stacked=True)
#    fig1.set_title('')
#    fig1.set_ylabel(var)
    
#    plt.subplot(1, 2, 2)
#    fig2 = data.groupby([var])['QuoteConversion_Flag'].mean().plot.bar()
    #fig.set_ylabel('% of Conversions')
    #fig.set_xlabel(var)
#    plt.show()


# ## Split the data

# ### Get the variables types updated

# make list of variables  types
dates = [var for var in data.columns if 'Date' in var or 'Year' in var]


# numerical: discrete vs continuous
discrete = [var for var in data.columns if data[var].dtype!='O' and var!='label' and 2< data[var].nunique()<15 and var not in dates]
continuous = [var for var in data.columns if data[var].dtype!='O' and var!='label' and var not in discrete and var not in dates]
binary_num = [var for var in data.columns if data[var].nunique()==2 and  data[var].dtype!='O' and var!='label']
binary_cat = [var for var in data.columns if data[var].nunique()==2 and  data[var].dtype=='O' and var!='label']
# mixed
mixed = []

# categorical
categorical = [var for var in data.columns if data[var].dtype=='O' and var!='label' and var not in mixed and var not in binary_cat and var not in binary_num]

print('There are {} date variables'.format(len(dates)))
print('There are {} discrete variables'.format(len(discrete)))
print('There are {} binary categorical variables'.format(len(binary_cat)))
print('There are {} binary numeric variables'.format(len(binary_num)))
print('There are {} continuous variables'.format(len(continuous)))
print('There are {} categorical variables'.format(len(categorical)))
print('There are {} mixed variables'.format(len(mixed)))


# # Separate into train and test set

# Let's separate into train and test set
X_train, X_test, y_train, y_test = train_test_split(data.drop(['labels'], axis=1),
                                                    data['labels'],
                                                    test_size=0.3,
                                                    random_state=0)

X_train.shape, X_test.shape


X_train.head()


# ### Save datasets

# let's now save the train and test sets for the next notebook!

X_train.to_csv(data_folder /'processed/xtrain.csv', index=False, header=True)
X_test.to_csv(data_folder /'processed/xtest.csv', index=False, header=True)

y_train.to_csv(data_folder /'processed/ytrain.csv', index=False, header=True)
y_test.to_csv(data_folder /'processed/ytest.csv', index=False, header=True)


# missing inputatio for discrete 

# Rare value encoder
#rare_encoder = RareLabelCategoricalEncoder(
#    tol=0.05,  # minimal percentage to be considered non-rare
#    n_categories=5, # minimal number of categories the variable should have to re-cgroup rare categories
#    variables=categorical # variables to re-group
#)  


#rare_encoder.fit(tmp2)


#rare_encoder.variables


# the encoder_dict_ is a dictionary of variable: frequent labels pair
#rare_encoder.encoder_dict_


#tmp22 = rare_encoder.transform(tmp2)


# SAme but with a pipeline

# # Build Pipeline

#quote_pipe = Pipeline([
#    # Add missing indicator
#    ('missing_ind_categorical', mdi.AddMissingIndicator(variables=vars_with_na)),
#    # Impute categorical variables with value "Missing"
#    ('imputer_cat_miss',mdi.CategoricalVariableImputer(variables=categorical)),
#    ('imputer_cat_rare',ce.RareLabelCategoricalEncoder(tol=0.05, n_categories=5, variables=categorical)),
#    ('categorical_enc', ce.OneHotCategoricalEncoder(top_categories=None,drop_last=True, variables=categorical+binary_cat)),
#])


#quote_pipe.fit(X_train)

#X_train_transformed = quote_pipe.transform(X_train)
#X_test_transformed = quote_pipe.transform(X_test)
#X_train_transformed.head()


#X_train_transformed.dtypes


# ### Remove constant features

# remove constant features
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]

X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

X_train.shape, X_test.shape


constant_features


# ### Remove quasi-constant features

# remove quasi-constant features
#sel = VarianceThreshold(threshold=0.01)  # 0.1 indicates 99% of observations approximately

#sel.fit(X_train_transformed)  # fit finds the features with low variance

#sum(sel.get_support()) # how many not quasi-constant?


#features_to_drop = X_train_transformed.columns[[not i for i in sel.get_support() ]]
#features_to_drop


#X_train_transformed.drop(labels=features_to_drop, axis=1, inplace=True)
#X_test_transformed.drop(labels=features_to_drop, axis=1, inplace=True)

#X_train_transformed.head()


# ### Remove duplicated features

# check for duplicated features in the training set
duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            print(f'the values on {col_1} are the same as {col_2}')
            duplicated_feat.append(col_2)
            
len(duplicated_feat)


duplicated_feat = list(dict.fromkeys(duplicated_feat)) #remove duplicates
duplicated_feat


# remove duplicated features
X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

X_train.shape, X_test.shape


# ### Remove correlated features

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

current_continuos = intersection(continuous,X_train.columns)
current_continuos


# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train[current_continuos], 0.8)
print('correlated features: ', len(set(corr_features)))


X_train[current_continuos].corr()


#X_train_transformed.drop(labels=corr_features, axis=1, inplace=True)
#X_test_transformed.drop(labels=corr_features, axis=1, inplace=True)

#X_train_transformed.shape, X_test_transformed.shape


# ## Random Forest importance

# here I will do the model fitting and feature selection
# altogether in one line of code

# first I specify the Random Forest instance, indicating
# the number of trees

# Then I use the selectFromModel object from sklearn
# to automatically select the features

# SelectFrom model will select those features which importance
# is greater than the mean importance of all the features
# by default, but you can alter this threshold if you want to

sel_ = SelectFromModel(RandomForestClassifier(n_estimators=300),threshold="0.55*mean")
sel_.fit(X_train, y_train)


# this command let's me visualise those features that were selected.

# sklearn will select those features which importance values
# are greater than the mean of all the coefficients.

selected_feat=list(X_train.columns[sel_.get_support()])
selected_feat


# and now, let's compare the  amount of selected features
# with the amount of features which importance is above the
# mean importance, to make sure we understand the output of
# sklearn

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients greater than the mean coefficient: {}'.format(
    np.sum(sel_.estimator_.feature_importances_ > sel_.estimator_.feature_importances_.mean())))


X_train_model=X_train[selected_feat]
X_test_model=X_test[selected_feat]


# ### Bayesian Optimization

def objective(space):

    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    classifier = xgb.XGBClassifier(n_estimators = space['n_estimators'],
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            gamma = space['gamma'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree']
                            )
    
    classifier.fit(X_train_model, y_train)

    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train_model, y = y_train, cv = 5,scoring='roc_auc')
    CrossValMean = accuracies.mean()

    print("CrossValMean:", CrossValMean)

    return{'loss':1-CrossValMean, 'status': STATUS_OK }



space = {
    'max_depth' : hp.choice('max_depth', range(2, 9, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(100, 600, 25)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            return_argmin=False,
            trials=trials)

print("Best: ", best)


best


import json
# A method for saving object data to JSON file
def save_json(self, filepath):
    dict_ = {}
    dict_ = self

    # Creat json and save to file
    json_txt = json.dumps(dict_, indent=4)
    with open(filepath, 'w') as file:
        file.write(json_txt)
        
# A method for loading data from JSON file
def load_json(filepath):
    with open(filepath, 'r') as file:
        dict_ = json.load(file)

    return dict_


save_json(best,models_folder/'research_hyperparameters.json')


def create_schema(data, verbose=False):
    schema = {}
    for feature in data.columns:
        if verbose:
            print(f"----- {feature} ----")
            print(f"{data[feature].mean()}")
            print(f"{data[feature].std()}")
            print(f"{data[feature].min()}")
            print(f"{data[feature].max()}")
            print(f"{data[feature].unique()[0:20]}")
            print(f"-----           ----")
        thisdict = {"mean": float(data[feature].mean()),
                    "std": float(data[feature].std()),
                    "min": float(data[feature].min()),
                    "max": float(data[feature].max()),
                    "values": (data[feature].unique()[0:20]).tolist(),
                    "pct_miss": float(np.round(data[feature].isnull().mean(), 3)),
                    "type": str(data[feature].dtypes)
                   }
        schema[feature]=thisdict
    return schema


schema= create_schema(X_train_model)

save_json(schema,models_folder/'train_schema.json')
schema


best= load_json(models_folder/'research_hyperparameters.json')
best


train_schema= load_json(models_folder/'train_schema.json')
selected_feat=list(train_schema.keys())
selected_feat


X_train=pd.read_csv(data_folder /'processed/xtrain.csv')
X_test=pd.read_csv(data_folder /'processed/xtest.csv')
X_train_model=X_train[selected_feat]
X_test_model=X_test[selected_feat]                 

y_train=pd.read_csv(data_folder /'processed/ytrain.csv').values.ravel()
y_test=pd.read_csv(data_folder /'processed/ytest.csv').values.ravel()
#y_train.value()

#X_train_model.head()


# ### Fit Final Model

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = best['n_estimators'],
                            max_depth = best['max_depth'],
                            learning_rate = best['learning_rate'],
                            gamma = best['gamma'],
                            min_child_weight = best['min_child_weight'],
                            subsample = best['subsample'],
                            colsample_bytree = best['colsample_bytree']
                            )

classifier.fit(X_train_model, y_train,)

# Applying k-Fold Cross Validation  m
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_test_model, y = y_test, cv = 10,scoring='roc_auc')
CrossValMean = accuracies.mean()
print("Final CrossValMean: ", CrossValMean)

CrossValSTD = accuracies.std()
print("Final CrossValSTD: ", CrossValSTD)


## let's make predictions
X_train_preds_best = classifier.predict_proba(X_train_model,)[:,1]
X_test_preds_best = classifier.predict_proba(X_test_model,)[:,1]


print('Train set')
print('XGB roc-auc: {}'.format(roc_auc_score(y_train, X_train_preds_best)))

print('Test set')
print('XGB roc-auc: {}'.format(roc_auc_score(y_test, X_test_preds_best)))


# ###### Save and Load the models at this stage

from joblib import dump, load
model = load(models_folder/'research_model_v1.joblib') 
model


from joblib import dump, load
classifier = load(models_folder/'research_model_v1.joblib') 
classifier


# ### AUC confidence Interval

def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

statistics = bootstrap_auc(y_test, X_test_preds_best, ["Churn"])


def print_confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df

print_confidence_intervals(["Churn"],statistics)


from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred,class_labels):
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y, pred, n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()
    
plot_calibration_curve(y_test, X_test_preds_best,["Churn"])


# ### Calibration with Plat Scaling

from sklearn.linear_model import LogisticRegression as LR 

pred_calibrated = np.zeros_like(X_test_preds_best)
lr = LR(solver='liblinear', max_iter=10000)
lr.fit(X_test_preds_best.reshape(-1, 1), y_test)    
pred_calibrated = lr.predict_proba(X_test_preds_best.reshape(-1, 1))[:,1]
    

plot_calibration_curve(y_test, pred_calibrated,["label"])


statistics_calib = bootstrap_auc(y_test, pred_calibrated, ["Churn"])
print_confidence_intervals(["Churn"],statistics_calib)


#from sklearn.isotonic import IsotonicRegression as IR

#pred_calibrated = np.zeros_like(X_test_preds)
#ir = IR( out_of_bounds = 'clip' )
#ir.fit( X_test_preds.reshape(-1, 1), y_test)
#p_calibrated = ir.fit(X_test_preds.reshape(-1, 1))[:,1]


# ### AUC and PR Curves

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
def get_curve(gt, pred, target_names, curve='roc'):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt, pred)
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt, pred)
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt, pred)
            #avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(gt, pred, average='weighted')
            average_precision = average_precision_score(gt, pred)
            label = target_names[i] + " Avg. Precision: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)


get_curve(y_test, X_test_preds_best,["Churn"], curve='prc')


get_curve(y_test, X_test_preds_best,["Churn"], curve='roc')


# ### Feature Importance

# plot feature importance
plot_importance(classifier)
plt.show()


# plot feature importance
(classifier.feature_importances_ >0).sum()


import shap


# ## Model Explanation

# ### SHAP Values

import shap
shap.initjs()
explainer = shap.TreeExplainer(classifier,model_output="probability",
                               feature_perturbation ="interventional",
                               data=shap.sample(X_test_model,100))
shap_values = explainer.shap_values(X_test_model)
shap_interaction_values = shap.TreeExplainer(classifier).shap_interaction_values(X_test_model)

if isinstance(shap_interaction_values, list):
    shap_interaction_values = shap_interaction_values[1]
    
print('Expected Value:', explainer.expected_value)
pd.DataFrame(shap_values).head()

#https://slundberg.github.io/shap/notebooks/plots/decision_plot.html
#https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba


shap.initjs()
shap.force_plot(explainer.expected_value, 
                shap_values[0], X_test_model.iloc[0,:])



shap.decision_plot(
    base_value=explainer.expected_value,
    shap_values= shap_values[0,:],
    features=X_test_model.iloc[0,:],
    feature_names=X_test_model.columns.tolist(),
    link="identity",
    feature_order='importance'
    #new_base_value=0.5,
)


shap.force_plot(explainer.expected_value, 
                shap_values[:1000,:], X_test_model.iloc[:1000,:])


shap.summary_plot(shap_values, 
                  X_test_model, plot_type="bar")


shap.summary_plot(shap_values, X_test_model)


order_features = np.argsort(-np.sum(np.abs(shap_values),0))
X_test_model.columns[order_features]


for i in range(len(order_features)):
    print(f"---- Shap plot : {X_test_model.columns[order_features][i]} ------")
    shap.dependence_plot(X_test_model.columns[order_features[i]], shap_values, X_test_model)


#shap_interaction_values = shap.TreeExplainer(classifier).shap_interaction_values(X_test_model)


shap.summary_plot(shap_interaction_values, X_test_model)


shap.decision_plot(
    base_value=explainer.expected_value,
    shap_values= shap_interaction_values[0,:],
    features=X_test_model.iloc[0,:],
    feature_names=X_test_model.columns.tolist(),
    link="identity",
    feature_order='importance'
    #new_base_value=0.5,
)


shap.dependence_plot(ind='feature_3', interaction_index='feature_3',
                     shap_values=shap_values, 
                     features=X_test_model,  
                     display_features=X_test_model)


import matplotlib.pylab as pl

tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
pl.figure(figsize=(12,12))
pl.imshow(tmp2)
pl.yticks(range(tmp2.shape[0]), X_test_model.columns[inds], rotation=50.4, horizontalalignment="right")
pl.xticks(range(tmp2.shape[0]), X_test_model.columns[inds], rotation=50.4, horizontalalignment="left")
pl.gca().xaxis.tick_top()
pl.show()


from sklearn.inspection import partial_dependence

def plot_pdp(model, X, feature, target=False, return_pd=False, y_pct=True, figsize=(10,9), norm_hist=True, dec=.5):
    # Get partial dependence
    pardep = partial_dependence(model, X, [feature])
    
    # Get min & max values
    xmin = pardep[1][0].min()
    xmax = pardep[1][0].max()
    ymin = pardep[0][0].min()
    ymax = pardep[0][0].max()
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.grid(alpha=.5, linewidth=1)
    
    # Plot partial dependence
    color = 'tab:blue'
    ax1.plot(pardep[1][0], pardep[0][0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel(feature, fontsize=14)
    
    tar_ylabel = ': {}'.format(target) if target else ''
    ax1.set_ylabel('Partial Dependence{}'.format(tar_ylabel), color=color, fontsize=14)
    
    tar_title = target if target else 'Target Variable'
    ax1.set_title('Relationship Between {} and {}'.format(feature, tar_title), fontsize=16)
    
    if y_pct and ymin>=0 and ymax<=1:
        # Display yticks on ax1 as percentages
        fig.canvas.draw()
        labels = [item.get_text() for item in ax1.get_yticklabels()]
        labels = [int(np.float(label.replace('−', '-'))*100) for label in labels]
        labels = ['{}%'.format(label) for label in labels]
        ax1.set_yticklabels(labels)
    
    # Plot line for decision boundary
    ax1.hlines(dec, xmin=xmin, xmax=xmax, color='black', linewidth=2, linestyle='--', label='Decision Boundary')
    ax1.legend()

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.hist(X[feature], bins=80, range=(xmin, xmax), alpha=.25, color=color, density=norm_hist)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Distribution', color=color, fontsize=14)
    
    if y_pct and norm_hist:
        # Display yticks on ax2 as percentages
        fig.canvas.draw()
        labels = [item.get_text() for item in ax2.get_yticklabels()]
        labels = [int(np.float(label.replace('−', '-'))*100) for label in labels]
        labels = ['{}%'.format(label) for label in labels]
        ax2.set_yticklabels(labels)

    plt.show()
    
    if return_pd:
        return pardep


from sklearn.inspection import partial_dependence
for i in range(len(order_features)):
    print(f"---- PDP plot : {X_test_model.columns[order_features][i]} ------")
    plot_pdp(classifier, X_test_model, X_test_model.columns[order_features[i]], target='Churn')


# ### ELI5

import eli5
eli5.show_weights(classifier.get_booster())


doc_num = 100

predictions=X_test_preds_best
gt=y_test

print('Actual Label:', gt[doc_num])
print('Predicted Label:', predictions[doc_num])
eli5.show_prediction(classifier.get_booster(), X_test_model.iloc[doc_num], 
                     feature_names=list(X_test_model.columns),
                     show_feature_values=True)


X_test_preds_best[100]


# ### Save Features

# now we save the selected list of features
pd.Series(selected_feat).to_csv(models_folder /'research_selected_features_classifier.csv', index=False)


# ### Save seed

# # Adding custom transformer for sklearn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline 

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 


#Custom Transformer that store a schema of data inside teh pipeline
class SchemaBuild( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
        self._feature_schema ={}
    
    #Return self with a data schema added
    def fit( self, x, y = None ):
        for feature in x.columns:
            thisdict = {"mean": float(x[feature].mean()),
                        "std": float(x[feature].std()),
                        "min": float(x[feature].min()),
                        "max": float(x[feature].max()),
                        "values": (x[feature].unique()[0:20]).tolist(),
                        "pct_miss": float(np.round(x[feature].isnull().mean(), 3)),
                        "type": str(x[feature].dtypes)
                       }
            self._feature_schema[feature]=thisdict
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, x, y = None ):
        return x


train_schema= load_json(models_folder/'train_schema.json')
features=list(train_schema.keys())


churn_pipe = Pipeline([
    # Add missing indicator
    ('feature_selector',FeatureSelector(features)),
    ('feature_schema',SchemaBuild(features))])

churn_pipe.fit(X_train)

X_train_model = churn_pipe.transform(X_train)
X_test_model = churn_pipe.transform(X_test)


#X_train_model.columns
#churn_pipe['feature_schema']._feature_schema


hyperparameters= load_json(models_folder/'research_hyperparameters.json')
hyperparameters


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = hyperparameters['n_estimators'],
                            max_depth = hyperparameters['max_depth'],
                            learning_rate = hyperparameters['learning_rate'],
                            gamma = hyperparameters['gamma'],
                            min_child_weight = hyperparameters['min_child_weight'],
                            subsample = hyperparameters['subsample'],
                            colsample_bytree = hyperparameters['colsample_bytree'],
                            random_state=0
                            )

model.fit(X_train_model, y_train)


from joblib import dump, load
dump(model, models_folder/'research_model_v1.joblib') 


from joblib import dump, load
model = load(models_folder/'research_model_v1.joblib') 
model


churn_pipe = Pipeline([
    # Add missing indicator
    ('feature_selector',FeatureSelector(features)),
    ('feature_schema',SchemaBuild(features)),
    ('model_xgb',XGBClassifier(**hyperparameters))])

churn_pipe.fit(X_train,y_train)

churn_pipe


## let's make predictions
X_train_preds_best = churn_pipe.predict_proba(X_train_model,)[:,1]
X_test_preds_best = churn_pipe.predict_proba(X_test_model,)[:,1]

print('Train set')
print('XGB roc-auc: {}'.format(roc_auc_score(y_train, X_train_preds_best)))

print('Test set')
print('XGB roc-auc: {}'.format(roc_auc_score(y_test, X_test_preds_best)))


churn_pipe['feature_schema'].


# ### Save Pipeline

from joblib import dump, load
dump(churn_pipe, models_folder/'research_feature_extractor_v1.joblib') 


# ##### Load ans test Pipeline

pipeline = load(models_folder/'research_feature_extractor_v1.joblib') 
pipeline


## let's make predictions
X_train_preds_best = pipeline.predict_proba(X_train_model,)[:,1]
X_test_preds_best = pipeline.predict_proba(X_test_model,)[:,1]

print('Train set')
print('XGB roc-auc: {}'.format(roc_auc_score(y_train, X_train_preds_best)))

print('Test set')
print('XGB roc-auc: {}'.format(roc_auc_score(y_test, X_test_preds_best)))


pipeline['feature_schema']._feature_schema['feature_0']


pipeline['feature_selector']._feature_names




