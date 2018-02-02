""" Module that creates a dictionary for each CAWS site to be
used in the scikit-learn processing. """

import numpy as np
import pandas as pd


def load_caws_site(site, file_path):
    """
    Does the first pass of GBM on the CAWS site to determine which
    variables to eliminate from initial analysis.

    Parameters
    ----------
    site: str
        Dictionary re
    num_ensembles: int
        Number of times to run GBM for bootstrap approach.

    Return
    ------
    site_dict: dict
        Dictionary of values describing site.

    """

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
    # Solar radiation, air temperature, water temperature,
    # discharge, stage, combined sewer outflows (CSOs),
    # turbidity, suspended solids, pH, total organic carbon (TOC),
    # dissolved oxygen (DO), total dissolved solids (TDS),
    # total phosphorus (TP), total Kjeldahl nitrogen (TKN),
    # Chlorophyll (Chl), Chlorine (Cl), nitrate (NO3), ammonia (NH3),
    # sulfate (SO4), fluoride (F), heavy metals (need to specify).
    return_dic['feature_names'] = feature_names
    return_dic['data'] = data_list
    return_dic['target'] = site_df['Fecal'].values
    return_dic['DESCRIP'] = ('Dataset for ' + site)
    return return_dic
