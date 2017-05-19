# import sklearn BaseEstimator etc to use
import pandas as pd
import numpy as np
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin, RegressorMixin

# Activate R objects.
pandas2ri.activate()
R = ro.r
NULL = ro.NULL

# import the R packages required.
base = importr('base')

# https://cran.r-project.org/web/packages/ranger/ranger.pdf #
ranger = importr('ranger')


class RangerClassifier(BaseEstimator, ClassifierMixin):
    """
    From Ranger DESCRIPTION FILE:
    A fast implementation of Random Forests, particularly suited for high dimensional data.
    Ensembles of classification, regression, survival and probability prediction trees are supported.

    Because scikit-learn uses fit(X, y) we must join in R to make a data frame.  For this we HACK a variable called
    'RANGER_TARGET_DUMMY' to identify y to run against.

    We pull in all the options that ranger allows, but as this is a classifier we hard-code
    probability=True, to give probability values.
    
    We also eliminate the sample fraction option for ease. 
        sample_fraction = ifelse(replace,1, 0_632)
        
    """

    def __init__(self, formula='RANGER_TARGET_DUMMY~.', num_trees=500, mtry=NULL, importance="none",
                 write_forest=True, min_node_size=NULL, replace=True, case_weights=NULL, splitrule=NULL,
                 num_random_splits=1, alpha=0.5, minprop=0.1, split_select_weights=NULL,
                 always_split_variables=NULL, respect_unordered_factors=NULL, scale_permutation_importance=False,
                 keep_inbag=False, holdout=False, num_threads=1, save_memory=False, verbose=True, seed=NULL,
                 dependent_variable_name=NULL, status_variable_name=NULL, classification=NULL):
        self.formula = formula
        self.num_trees = num_trees
        self.mtry = mtry
        self.importance = importance
        self.write_forest = write_forest
        self.min_node_size = min_node_size
        self.replace = replace
        self.case_weights = case_weights
        self.splitrule = splitrule
        self.num_random_splits = num_random_splits
        self.alpha = alpha
        self.minprop = minprop
        self.split_select_weights = split_select_weights
        self.always_split_variables = always_split_variables
        self.respect_unordered_factors = respect_unordered_factors
        self.scale_permutation_importance = scale_permutation_importance
        self.keep_inbag = keep_inbag
        self.holdout = holdout
        self.num_threads = num_threads
        self.save_memory = save_memory
        self.verbose = verbose
        self.seed = seed
        self.dependent_variable_name = dependent_variable_name
        self.status_variable_name = status_variable_name
        self.classification = classification
        self.num_classes = None
        self.clf = None

    def fit(self, X, y):
        # First convert the X and y into a dataframe object to use in R
        # We have to convert the y back to a dataframe to join for using with R
        # We give the a meaningless name to allow the formula to work correctly.

        index_X = np.arange(X.shape[0])
        if isinstance(X, pd.DataFrame):
            index_X = X.index

        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=index_X)
        # Ensure the pd dataframe has the target column names with the dummy name.
        y.columns = ['RANGER_TARGET_DUMMY']
        # TODO have better handingly of numpy / pandas interfaces -> for V1 really should just have pandas for ease.


        self.num_classes = y.iloc[:, 0].nunique()
        r_dataframe = pd.concat([X, y], axis=1)
        r_dataframe['RANGER_TARGET_DUMMY'] = r_dataframe['RANGER_TARGET_DUMMY'].astype('str')
        self.clf = ranger.ranger(formula=self.formula,
                                 data=r_dataframe,
                                 num_trees=self.num_trees,
                                 mtry=self.mtry,
                                 importance=self.importance,
                                 write_forest=self.write_forest,
                                 min_node_size=self.min_node_size,
                                 probability=True,
                                 replace=self.replace,
                                 case_weights=self.case_weights,
                                 splitrule=self.splitrule,
                                 num_random_splits=self.num_random_splits,
                                 alpha=self.alpha,
                                 minprop=self.minprop,
                                 split_select_weights=self.split_select_weights,
                                 always_split_variables=self.always_split_variables,
                                 respect_unordered_factors=self.respect_unordered_factors,
                                 scale_permutation_importance=self.scale_permutation_importance,
                                 keep_inbag=self.keep_inbag,
                                 holdout=self.holdout,
                                 num_threads=self.num_threads,
                                 save_memory=self.save_memory,
                                 verbose=self.verbose,
                                 seed=self.seed,
                                 dependent_variable_name=self.dependent_variable_name,
                                 status_variable_name=self.status_variable_name,
                                 classification=self.classification)
        return

    def predict_proba(self, X):
        # Ranger doesnt have a specific separate predict and predict probabilities class, it is set in the params
        # REM: R is not Python :)
        pr = R.predict(self.clf, dat=X)
        pandas_preds = ro.pandas2ri.ri2py_dataframe(pr.rx('predictions')[0])
        if self.num_classes == 2:
            pandas_preds = pandas_preds.iloc[:, 1]
        return pandas_preds.values


class RangerRegressor(BaseEstimator, RegressorMixin):
    """
    From Ranger DESCRIPTION FILE:
    A fast implementation of Random Forests, particularly suited for high dimensional data.
    Ensembles of classification, regression, survival and probability prediction trees are supported.

    Because scikit-learn uses fit(X, y) we must join in R to make a data frame.  For this we HACK a variable called
    'RANGER_TARGET_DUMMY' to identify y to run against.

    We pull in all the options that ranger allows, but as this is a regressor we hard-code probability=False

    We also eliminate the sample fraction option for ease. 
        sample_fraction = ifelse(replace,1, 0_632)

    """

    def __init__(self, formula='RANGER_TARGET_DUMMY~.', num_trees=500, mtry=NULL, importance="none",
                 write_forest=True, min_node_size=NULL, replace=True, case_weights=NULL, splitrule=NULL,
                 num_random_splits=1, alpha=0.5, minprop=0.1, split_select_weights=NULL,
                 always_split_variables=NULL, respect_unordered_factors=NULL, scale_permutation_importance=False,
                 keep_inbag=False, holdout=False, num_threads=1, save_memory=False, verbose=True, seed=NULL,
                 dependent_variable_name=NULL, status_variable_name=NULL, classification=NULL):
        self.formula = formula
        self.num_trees = num_trees
        self.mtry = mtry
        self.importance = importance
        self.write_forest = write_forest
        self.min_node_size = min_node_size
        self.replace = replace
        self.case_weights = case_weights
        self.splitrule = splitrule
        self.num_random_splits = num_random_splits
        self.alpha = alpha
        self.minprop = minprop
        self.split_select_weights = split_select_weights
        self.always_split_variables = always_split_variables
        self.respect_unordered_factors = respect_unordered_factors
        self.scale_permutation_importance = scale_permutation_importance
        self.keep_inbag = keep_inbag
        self.holdout = holdout
        self.num_threads = num_threads
        self.save_memory = save_memory
        self.verbose = verbose
        self.seed = seed
        self.dependent_variable_name = dependent_variable_name
        self.status_variable_name = status_variable_name
        self.classification = classification
        self.clf = None

    def fit(self, X, y):
        # First convert the X and y into a dataframe object to use in R
        # We have to convert the y back to a dataframe to join for using with R
        # We give the a meaningless name to allow the formula to work correctly.

        index_X = np.arange(X.shape[0])
        if isinstance(X, pd.DataFrame):
            index_X = X.index

        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=index_X)
        # Ensure the pd dataframe has the target column names with the dummy name.
        y.columns = ['RANGER_TARGET_DUMMY']

        r_dataframe = pd.concat([X, y], axis=1)
        r_dataframe['RANGER_TARGET_DUMMY'] = r_dataframe['RANGER_TARGET_DUMMY'].astype('str')
        self.clf = ranger.ranger(formula=self.formula,
                                 data=r_dataframe,
                                 num_trees=self.num_trees,
                                 mtry=self.mtry,
                                 importance=self.importance,
                                 write_forest=self.write_forest,
                                 min_node_size=self.min_node_size,
                                 probability=False,
                                 replace=self.replace,
                                 case_weights=self.case_weights,
                                 splitrule=self.splitrule,
                                 num_random_splits=self.num_random_splits,
                                 alpha=self.alpha,
                                 minprop=self.minprop,
                                 split_select_weights=self.split_select_weights,
                                 always_split_variables=self.always_split_variables,
                                 respect_unordered_factors=self.respect_unordered_factors,
                                 scale_permutation_importance=self.scale_permutation_importance,
                                 keep_inbag=self.keep_inbag,
                                 holdout=self.holdout,
                                 num_threads=self.num_threads,
                                 save_memory=self.save_memory,
                                 verbose=self.verbose,
                                 seed=self.seed,
                                 dependent_variable_name=self.dependent_variable_name,
                                 status_variable_name=self.status_variable_name,
                                 classification=self.classification)
        return

    def predict(self, X):
        # Return the predicted vales
        pr = R.predict(self.clf, dat=X)
        pandas_preds = ro.pandas2ri.ri2py_dataframe(pr.rx('predictions')[0])
        return pandas_preds.values