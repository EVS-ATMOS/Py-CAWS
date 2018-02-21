""" Module for doing gradient boosting machine learning on the Py-CAWS data. """

from copy import deepcopy

import numpy as np
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score


def get_gbm_importance(site_dict, num_ensembles=250, target_name='target'):
    """
    Does the first pass of GBM on the CAWS site to determine which variables
    to eliminate from initial analysis.

    Parameters
    ----------
    site_dict : dict
        Dictionary of values describing site returned from load_caws_site.
    num_ensembles : int
        Number of times to run GBM for bootstrap approach.
    target_name : str
        The name of the target variable.

    Returns
    -------
    site_dict : dict
        Dictionary of values describing site with 3 extra fields:
        five_importance = fifth percentile of importance
        med_importance = median of importance
        ninety_five_importance = 95th percentile of importance

    """

    X, y = shuffle(site_dict['data'], site_dict[target_name])
    num_features = X.shape[1]
    importance = np.zeros((num_ensembles, num_features))
    for rand_seeds in range(0, num_ensembles):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=rand_seeds)
        # The maximum depth for each tree can be arbitrarily set to 4.
        # The minimum sample split is 2, learning rate = 0.01, and the
        # loss method will the least squares.
        GBM = ensemble.GradientBoostingRegressor(
            min_samples_split=2, n_estimators=7500, learning_rate=0.01,
            max_depth=4)
        GBM.fit(X_train, y_train)
        feature_importance = GBM.feature_importances_
        # Make importances relative to max importance.
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        importance[rand_seeds] = feature_importance
        if rand_seeds % 10 == 0:
            print(str(rand_seeds) + ' ensembles done')

    site_dict['fifth_percentile_importance'] = np.percentile(
        importance, axis=0, q=5)
    site_dict['first_quartile_importance'] = np.percentile(
        importance, axis=0, q=25)
    site_dict['median_importance'] = np.percentile(
        importance, axis=0, q=50)
    site_dict['third_quartile_importance'] = np.percentile(
        importance, axis=0, q=75)
    site_dict['ninety_fifth_percentile_importance'] = np.percentile(
        importance, axis=0, q=95)
    del X_train, X_test, y_train, y_test
    return site_dict


def dimension_reduced_dict(
        site_dict, importance_field='ninety_fifth_percentile_importance',
        threshold=20):
    where_met = np.where(site_dict[importance_field] >= threshold)[0]
    return_dic = {}
    return_dic['feature_names'] = [site_dict['feature_names'][x] for x in where_met]
    return_dic['data'] = site_dict['data'][:, where_met]
    return_dic['target'] = site_dict['target']
    if site_dict['transformed'] is True:
        return_dic['transformed_target'] = site_dict['transformed_target']
    return_dic['DESCRIP'] = site_dict['DESCRIP']
    return_dic['transformed'] = site_dict['transformed']
    return return_dic


def transform_target(site_dict, target_name='target',
                     transformed_target_name='transformed_target'):
    """
    Transforms the target variable by taking the base 10 logarithm.

    Parameters
    ----------
    site_dict : dict
        Dictionary of values describing site from load_caws_site.
    target_name : str
        Name of target variable.
    transformed_target_name : str
        Name of transformed target variable.

    Returns
    -------
    site_dict : dict
        Site dictionary with transformed variable.

    """

    if not target_name in site_dict.keys():
        print(target_name + ' does not exist in dictionary!')
        return

    if site_dict['transformed'] is False:
        site_dict[transformed_target_name] = np.log10(site_dict[target_name])
        site_dict['transformed'] = True
    else:
        print('GBM values already transformed!')

    return site_dict


def do_GBM_second_pass(site_dict, n_variables=15, target_name='target',
                       n_estimators=7500,
                       importance_variable='median_importance',
                       cv_folds=5):
    """
    Does the second pass on GBM using only the top n variables. Performs
    error analysis and cross validation analysis.

    Parameters
    ----------
    site_dict : dict
        Dictionary of values with importances added in.
    n_variables : int
        Number of variables to include in final dictionary.
    target_name : str
        Name of target variable.
    n_estimators : int
        Number of estimators for GBM. Default is 7500.
    cv_fols : int
        Number of cross validation folds. Default is 5.

    Returns
    -------
    gbm_dict : dict
        Dictionary with importances, RMSE, deviances, etc.

    """

    new_dict = deepcopy(site_dict)
    importance_sorted = sorted(new_dict[importance_variable], reverse=True)
    new_dict = dimension_reduced_dict(
        new_dict, importance_variable,
        threshold=importance_sorted[n_variables-1])
    X, y = shuffle(new_dict['data'], new_dict[target_name])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=3)

    # The maximum depth for each tree can be arbitrarily set to 4.
    # The minimum sample split is 2, learning rate = 0.01, and the
    # loss method will the least squares.
    GBM = ensemble.GradientBoostingRegressor(min_samples_split=2,
                                             n_estimators=n_estimators,
                                             learning_rate=0.01,
                                             max_depth=4)
    GBM.fit(X_train, y_train)
    feature_importance = GBM.feature_importances_

    # Make importances relative to max importance.
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    test_score = np.zeros((GBM.n_estimators), dtype=np.float64)
    for i, y_pred in enumerate(GBM.staged_predict(X_test)):
        test_score[i] = GBM.loss_(y_test, y_pred)

    new_dict['num_estimators'] = 7500
    new_dict['training_deviance'] = GBM.train_score_
    new_dict['test_deviance'] = test_score
    new_dict['test_mse'] = mean_squared_error(y_test, GBM.predict(X_test))
    new_dict['training_mse'] = mean_squared_error(
        y_train, GBM.predict(X_train))
    new_dict['test_rmse'] = np.sqrt(new_dict['test_mse'])
    new_dict['training_rmse'] = np.sqrt(new_dict['training_mse'])
    new_dict['regressor'] = GBM
    new_dict['test_r2'] = r2_score(y_test, GBM.predict(X_test))
    new_dict['training_r2'] = r2_score(y_train, GBM.predict(X_train))

    if new_dict['transformed'] is False:
        new_dict['y_test_true'] = y_test
        new_dict['y_train_true'] = y_train
        new_dict['y_true'] = y
        new_dict['test_predict'] = GBM.predict(X_test)
        new_dict['train_predict'] = GBM.predict(X_train)
    else:
        new_dict['y_test_true'] = np.power(10, y_test)
        new_dict['y_train_true'] = np.power(10, y_train)
        new_dict['y_true'] = np.power(10, y)
        new_dict['test_predict'] = np.power(10, GBM.predict(X_test))
        new_dict['train_predict'] = np.power(10, GBM.predict(X_train))

    cross_val = cross_val_predict(GBM, X, y, cv=5)
    new_dict['cv_mse'] = mean_squared_error(y, cross_val)
    new_dict['cv_rmse'] = np.sqrt(new_dict['test_mse'])
    new_dict['cv_r2'] = r2_score(y, cross_val)
    if new_dict['transformed'] is False:
        new_dict['cv_predict'] = cross_val
    else:
        new_dict['cv_predict'] = np.power(10, cross_val)

    return new_dict


def predict_GBM(site_dict, predict_dict):
    """
    Does the second pass on GBM using only the top n variables. Performs
    error analysis and cross validation analysis.

    Parameters
    ----------
    site_dict : dict
        Dictionary of values from second pass of GBM.
    predict_dict : dict
        Dictionary of values from load_predict_site.

    Returns
    -------
    predict_dict : dict
        Dictionary with added prediction of FIB.

    """

    # Reduce dimensionality of predicted dictionary.
    feature_names = site_dict['feature_names']
    data_list = []
    for i in range(len(predict_dict['feature_names'])):
        if predict_dict['feature_names'][i] in site_dict['feature_names']:
            data_list.append(predict_dict['data'][:, i])

    data_list = np.stack(data_list)
    data_list = np.transpose(data_list)
    fecal_predict = site_dict['regressor'].predict(data_list)
    if site_dict['transformed'] is True:
        predict_dict['predicted_target'] = np.power(10, fecal_predict)
    else:
        predict_dict['predicted_target'] = fecal_predict
    return predict_dict
