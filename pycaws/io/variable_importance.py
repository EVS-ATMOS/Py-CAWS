""" Module for reading CAWS data into DataFrames and defines variable names
for the data in each DataFrame to then be processed by GBM. """

import numpy as np
import pandas as pd


def load_caws_site(site, file_path):
    """ Return a Pandas DataFrame with the site data. """
    file_name = file_path + '/' + site + '.xlsx'
    print('Loading ' + file_name)
    site_df = pd.read_excel(file_name)

    print('Generating dictionary...')
    var_list = list(site_df.columns)

    # Omit first 2 and last entry of xlsx file
    feature_names = var_list[2:-1]
    # feature_names = ['DOMan', 'TwMan', 'pH', 'NO3']
    data_list = np.column_stack(
        [site_df[feature].values for feature in feature_names])
    return_dic = {}
    # Solar radiation, air temperature, water temperature, discharge, stage,
    # combined sewer outflows (CSOs), turbidity, suspended solids, pH, total
    # organic carbon (TOC), dissolved oxygen (DO), total dissolved solids
    # (TDS), total phosphorus (TP), total Kjeldahl nitrogen (TKN), Chlorophyll
    # (Chl), Chlorine (Cl), nitrate (NO3), ammonia (NH3), sulfate (SO4),
    # fluoride (F), heavy metals (need to specify).
    return_dic['feature_names'] = feature_names
    return_dic['data'] = data_list
    return_dic['target'] = site_df['Fecal'].values
    return_dic['DESCRIP'] = ('Dataset for ' + site)
    return_dic['transformed'] = False
    return return_dic


def load_predict_site(site, file_path):
    """ Returns a Pandas DataFrame with the prediction site data. """
    file_name = file_path + '/' + site + '_Predict.xlsx'
    print('Loading ' + file_name)
    site_df = pd.read_excel(file_name)
    print('Generating dictionary...')
    var_list = list(site_df.columns)

    feature_names = var_list[2:]
    #feature_names = ['DOMan', 'TwMan', 'pH', 'NO3']

    data_list = np.column_stack(
        [site_df[feature].values for feature in feature_names])
    return_dic = {}
    # Solar radiation, air temperature, water temperature, discharge, stage,
    # combined sewer outflows (CSOs), turbidity, suspended solids, pH, total
    # organic carbon (TOC), dissolved oxygen (DO), total dissolved solids (TDS),
    # total phosphorus (TP), total Kjeldahl nitrogen (TKN), Chlorophyll (Chl),
    # Chlorine (Cl), nitrate (NO3), ammonia (NH3), sulfate (SO4), fluoride (F),
    # heavy metals (need to specify).
    return_dic['feature_names'] = feature_names
    return_dic['data'] = data_list
    return_dic['DESCRIP'] = ('Prediction dataset for ' + site)
    return return_dic


def save_predict_site(predict_dict, site, out_file_name):
    """ Saves a prediction into an .xlsx file. """
    sites = {'SITE CODE': [site, site, site, site, site, site],
             'Prediction': ['I', 'II', 'III', 'IV', 'V', 'VI']}

    out_df = pd.DataFrame(sites)
    if 'predicted_target' in predict_dict.keys():
        out_df['FIB_predicted'] = predict_dict['predicted_target']

    columns = ['SITE CODE', 'Predicition', 'FIB_predicted']

    for i in range(len(predict_dict['feature_names'])):
        out_df[predict_dict['feature_names'][i]] = predict_dict['data'][:, i]
        columns.append(predict_dict['feature_names'][i])

    out_df.to_excel(out_file_name)
