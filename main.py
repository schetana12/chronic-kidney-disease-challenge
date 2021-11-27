"""
Performs two tasks on a given dataset:

TASK1: Get important features using any of the below implemented approaches:
            Approach1: Feature elimination using Correlation among (features, target)
            Approach2: Sklearn's SelectKBest(Not implemented)
            ... : <More approaches can be implemented as per future requirements>
       
TASK2: Get factors that can identify subtypes of the disease:
            Approach1: • Used PCA to identify the features that explain the maximum variance in the dataset
                       • Visualise and save the results of identified features for analysis by user
"""
__author__ = "Chetana Sharma"


import argparse
import arff
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing

from approaches import correlated_features, select_k_best
from visualise_utils import visualise_categories_pca, visualise_numerics_pca

# global thresholds
TARGET_THRESHOLD = 0.4
CODES_FOR_1 = ['yes', 'abnormal', 'present' ,'poor']
CODES_FOR_0 = ['no', 'normal', 'notpresent' ,'good']

method_to_func = {'corr':correlated_features, 'selectkbest': select_k_best}

class FeatureSelection():
    def __init__(self, args):
        self.args = args
        self.feature_selection_method = method_to_func[args['method']]
        self.features = []
        self.raw_data = None
        self.data = self.read_data()
        self.pca_imp_features = []
        
    def read_data(self):
        """
            Reads a file from filepath (currently supports only "arff" and "csv" formats) and returns a pandas dataframe
            TODO: Handle different file formats
        """
        
        filepath = self.args['filepath']
        extension = os.path.splitext(filepath)[1]

        if extension==".arff":
            dataset = arff.load(open(filepath, 'r'))
            headers = []
            for feature in dataset['attributes']:
                headers.append(feature[0])
            data = pd.DataFrame(dataset['data'], columns = headers)
        
        if extension==".csv":
            data = pd.read_csv(filepath)

        self.raw_data = data.copy()
        data[self.args['target']] = (data[self.args['target']] == self.args['postive_class_code']).astype('int')
        return data
    
    def impute_missing_data(self):
        """
            Identifies the missing/NA values in the data and imputes them.
            Categorical features are imputed with mode value
            Numerical features can be imputed with mean or median value depending on what was passed
        """
        for indx in self.data:
            feature = self.data[indx]
            if len(feature[feature.isnull()==True])==0:
                return
        
            # if the feature is NUMERIC
            if is_numeric_dtype(feature):
                if self.args['with_mean']:
                    mean_value = self.data[indx].mean()
                    self.data.loc[self.data[indx].isnull() == True, indx] = mean_value
                    continue
                else:
                    median_value = feature.median()
                    self.data.loc[self.data[indx].isnull() == True, indx] = median_value
                    continue

            # if the feature is CATEGORICAL
            mode_value = feature.mode()[0] # .mode() returns a Series this is why indexing
            self.data.loc[self.data[indx].isnull() == True, indx] = mode_value
        
    def encode_categorical_data(self):
        """
            Performs One-hot encoding for categorical variables
        """
        for indx in self.data:
            if not is_numeric_dtype(self.data[indx]):
                try:
                    self.data[indx] = pd.to_numeric(self.data[indx])
                except:
                    # if the feature is not numeric and also cannot be converted to a numeric type, 
                    # do the following
                    label_one_value = list(set(self.data[indx].unique()).intersection(CODES_FOR_1))[0]
                    self.data[indx] = (self.data[indx] == label_one_value).astype('int')

    
    def preprocess(self):
        """
            Does approach agnostic preprocessing
        """
        self.impute_missing_data()
        self.encode_categorical_data()

    def scale_data(self, X_train, X_test):
        """
            Performs Standard Scaling
        """
        scaler = StandardScaler().fit(X_train)
        Xtrain_scaled = scaler.transform(X_train)
        Xtest_scaled = scaler.transform(X_test)
        
        return Xtrain_scaled, Xtest_scaled
    

    def evaluate_features(self, X, y):
        """ 
            Evaluates the identified features by testing the metrics on a Random Forest Classifier
        """
        
        # performing train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=245, stratify=y)
                
        # scaling / normalizing / standardizing
        Xtrain_scaled, Xtest_scaled = self.scale_data(X_train, X_test)
        
        # fitting Random Forest classifier as the baseline model for comparision
        model = RandomForestClassifier(criterion='entropy', random_state=47)
        model.fit(Xtrain_scaled, y_train)
        
        # obtaining the evaluation metrics
        acc, roc, prec, rec, f1, r2 = self.get_metrics(model, Xtest_scaled, y_test)
        results_df = pd.DataFrame([[acc, roc, prec, rec, f1, r2, Xtest_scaled.shape[1]]], 
                            columns=["Accuracy", "ROC", "Precision", "Recall", "F1 Score", "R2",'Feature Count'])

        return results_df
    

    def get_metrics(self, model, X_test, y_test):
        """
            Calculates Accuracy, ROC, Precision, Recall, F1 and R2 metrics, given X_test and y_test
        """
        # get the predictions
        y_hat = model.predict(X_test)
        
        # calculate all the metrics
        acc = accuracy_score(y_test, y_hat)
        roc = roc_auc_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat)
        rec = recall_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        
        return acc, roc, prec, rec, f1, r2
    
    def numeric_category_features(self):
        """
            Identifies the categorical/nominal features from the dataset
        """

        numerics, categories = [], []
        for indx in self.data:
            if is_numeric_dtype(self.raw_data[indx]):
                numerics.append(indx)
            else:
                categories.append(indx)
        
        return numerics, categories
    
    def get_subtypes_pca(self):
        """ 
            Does PCA analysis and identifies the important features that explains the most variance in the dataset.
            Hence, identifies subtypes of the disease. 
        """
        # dropping the "target" column since thats not required to perform PCA
        X_pca = self.data.drop(labels=[self.args['target']], axis=1)
        pca_columns = X_pca.columns
        
        # do PCA
        X_pca = preprocessing.StandardScaler().fit_transform(X_pca)
        pca = PCA(n_components=2)
        pca.fit(X_pca)
        
        # finds the top 2 Principal components and the features that contribute the most to it        
        print('\n\n TASK 2: SUBTYPES ANALYSIS USING PCA')
        print(f'Explained variance by first two components of PCA: {pca.explained_variance_ratio_}')
        imp_indices = np.argsort(abs(pca.components_[0]))
        topk = int(len(imp_indices)*0.35)
        imp_features = [pca_columns[indx] for indx in imp_indices[-(topk):]]
        print(f'Features contributing to principal components: {imp_features}')
        print('\nRESULTS SAVED AT:- ./outputs/ folder')
        self.pca_imp_features = imp_features

        # store the plots on disk
        self.save_visual_analysis()

    def save_visual_analysis(self):
        """ 
            Visualises the categorical and numerical factors identified using PCA for subtype analysis.
        """
        
        numerics, categories = self.numeric_category_features()
        pca_features = self.pca_imp_features
        category_pca_features = list(set(pca_features).intersection(categories))
        numeric_pca_features = list(set(pca_features).intersection(numerics))

        visualise_categories_pca(self.raw_data, self.args['target'], category_pca_features, self.args['outpath'])
        visualise_numerics_pca(self.raw_data, self.args['target'], numeric_pca_features, self.args['outpath'])
    
    def compare_features(self):
        """
            Compares (results corresponding to "SELECTED features") VS (results corresponding to "ALL features")
        """
        # pre-processed dataset with all the features
        X = self.data.drop(self.args['target'], axis=1)
        y = self.data[self.args['target']]
        
        results_with_all_data = self.evaluate_features(X, y)
        results_with_all_data['method'] = 'all features'
        
        # dataset with just selected features
        X = self.data[self.features]
        results_with_features = self.evaluate_features(X, y)
        results_with_features['method'] = 'important features'
        
        # concatenating and displaying the results 
        print('\n\n TASK 1: IDENTIFY KEY FACTORS')
        print(f'The important features obtained are {self.features}')
        print(f'The results of a RandomForest Classifier on all features and important features are below:')
        print(pd.concat([results_with_all_data, results_with_features], axis=0))

    
    def get_important_features(self):
        """
            Driver function
        """

        # approach agnostic preprocessing
        self.preprocess()
        self.features = self.feature_selection_method(self.data)
        self.compare_features()

        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-fp', '--filepath',
                        type=str,
                        required=True,
                        help="Filepath - can be path on local disk or a URL")

    parser.add_argument('-out', '--outpath',
                        type=str,
                        default='./outputs',
                        help="Output Path - can be path on local disk where the analysis outputs need to be saved")

    parser.add_argument('-tg', '--target',
                        type=str,
                        default='class',
                        help="Target - name of the target variable in dataset")
    
    parser.add_argument('-pos', '--postive_class_code',
                        type=str,
                        default='ckd',
                        help="Positive class code - the class used to mark disease samples")
    
    parser.add_argument('-mt', '--method',
                        type=str,
                        default='corr',
                        choices=['corr', 'selectkbest'], 
                        help="Different approaches to find the important features. \
                        1. corr - Correlation method \
                        2. selectkbest - Using sklearn's SelectKBest")
                        #TODO: Add more feature selection methods)
    
    parser.add_argument('-wm', '--with_mean',
                        type=bool,
                        default=True,
                        help="Whether to impute missing numerical values with mean. If false, values are \
                            imputed with 'median' instead")

    args = vars(parser.parse_args())
    
    fs = FeatureSelection(args)
    
    # TASK 1: get important features
    fs.get_important_features()
    
    # TASK 2: gets important features based on PCA and visualizes features identifying the disease distictly
    fs.get_subtypes_pca()