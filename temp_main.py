import warnings
warnings.filterwarnings("ignore")

# basic package
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# feature selection package
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import boxcox
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV

# mean normalization
def mean_normalize(df):
    df_norm = (df-df.mean())/df.std()
    return df_norm

# min-max normalization
def minMax_normalize(df):
    df_norm = (df-df.min())/(df.max()-df.min())
    return df_norm

# normalize data to range [0, 1]
def zeroOne_normalize(df):
    col_name = list(df)
    df_norm = MinMaxScaler().fit_transform(df)
    df_norm_table = pd.DataFrame(df_norm, columns=col_name)
    return df_norm_table

# box-cox normalization
def boxcox(df, var):
    df_tranformed = df.copy()
    df_transformed[var] = boxcox(df_tranformed[var]+1)[0]
    return df_transformed 

# create polynomial features
# default degree: 2
def polyFeature(x, deg=2):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    poly = PolynomialFeatures(degree=deg).fit(x)
    x_poly = poly.transform(x_scaled)
    return x_poly

# univariance feature selection with polynomial features
# get the best number of features
# return feature names
def univariance(x, y):
    logreg = LogisticRegression(C=1)
    logreg.fit(x, y)
    scores = cross_val_score(logreg, x, y, cv=10)
#     print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    highest_score = np.mean(scores)
    
    # get polynomial features of degree 2
    x_poly = polyFeature(x)
    
    # get feature subset of all size
    # update best number of features 
    for i in range(1, x_poly.shape[1]+1, 1):
        select = SelectKBest(score_func=chi2, k=i)
        select.fit(x_poly, y)
        x_selected = select.transform(x_poly)
     
        logreg.fit(x_selected, y)
        scores = cross_val_score(logreg, x_selected, y, cv=10)
#         print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i, np.mean(scores), np.std(scores)))
         
        if np.mean(scores) > highest_score or (np.mean(scores) == highest_score and np.std(scores) < std):
            highest_score = np.mean(scores)
            std = np.std(scores)
            k_features_highest_score = i
            selected = x_selected  
    
    # get names of selected features
    selector = SelectKBest(score_func=chi2, k=k_features_highest_score)
    fit = selector.fit(x_poly, y)
    index = selector.get_support(indices=True) 
    
    print('Number of Features: %i' % k_features_highest_score)
    print("Indexes of Selected Features: " + str(index))         

# Recursive Feature Elimination
def RFE(x, y):   
    model = LogisticRegression()
    rfe = RFECV(model, step=10, min_features_to_select=20)
    fit = rfe.fit(x, y)
    selected = []
    for bool, feature in zip(fit.support_, list(x)):
        if bool:
            selected.append(feature)    
    
    print("Number of Features: " + str(fit.n_features_))
    print("Selected Features: " + str(selected))

# convert categorical features to numeric
def convertToNumeric(string):
    arr = string.split()
    if arr[0]=='Minimal':
        return 1
    elif arr[0]=='Mild':
        return 2
    elif arr[0]=='Moderately':
        return 3
    elif arr[0]=='Severe':
        return 4
    else: # na
        return 0

# get csv file in path, encode 'Severity' to numeric, select only numeric data
def cleanData(path):
    df = pd.read_csv(path, index_col=0)
    df['Severity_score'] = df['Severity'].apply(convertToNumeric)
    df = df.select_dtypes(['number']).dropna(axis=1,how='any')
    return df

# separate target class column from feature column 
def separateVars(df):
    y = df['Severity_score']
    x= df.drop(['Severity_score'], axis=1)
    return x, y

# select top 10 features with biggest correlation
def top10(df):
    numRow = len(df.rows)
    if numRow < 10: # small dataset: feature number < 10
        return df
    else: 
        return df[:10]

# rank features by correlation
def corrRanking(path):
    df = cleanData(path)
    y = df['Severity_score']
    x= df.drop(['Severity_score'], axis=1)
    featureList = []
    
    numFeature = len(x.columns)
    for i in range(0,numFeature):    
        colName = x.columns[i]
        col = x[colName]
        corr = col.corr(y)
        value = (colName, corr)
        featureList.append(value)
     
    featureList = sorted(featureList, key=lambda x: x[-1], reverse=True)
    output = pd.DataFrame(featureList, columns=['Feature', 'Correlation'])
    return top10(output)

def main():
    csv_path = None
    while True:
        csv_path = input("Please provide a path to a csv file: ")
        if os.path.exists(csv_path) == False:
            print("The path that you provided is incorrect. Please try again.")
        elif os.path.isfile(csv_path) == False:
            print("The path that you provided is not a file. Please try again.")
        elif csv_path.endswith('.csv') == False:
            print("The path that you provided is not a csv file. Please try again.")
        else:
            break  
    df = cleanData(csv_path)
    x, y = separateVars(df)
    x_norm = zeroOne_normalize(x)
    print("RFE: ")
    RFE(x_norm, y)
    print("Univariance: ")
    univariance(x_norm, y)
    
if __name__ == '__main__':
    main() 
