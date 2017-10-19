""" Code that takes raw stage data and does aggregation at different
time intervals. Processed data can then be saved as excel or csv. """

import os

import pandas as pd
import numpy as np


def stage_output(stage_file, to_excel=True, excel_filepath=None,
                 to_csv=False, csv_filepath=None):
    """
    Stage Output, takes in raw stage data and outputs aggregated
    stage data.

    Parameter
    ---------
    stage_file : str
        Filename of stage file to process.

    Optional Parameters
    -------------------
    to_excel : bool
        Choose whether or not to write the data to a excel file.
        Default is True.
    excel_filepath : str
        Filepath location to save excel file to. Default is
        None and the excel file is saved to user's home directory.
    to_csv : bool
        Choose whether or not to write the data to a csv file.
        Default is False.
    csv_filepath : str
        Filepath location to save csv file to. Default is
        None and the csv file is saved to user's home directory.

    """

    # Creates a raw dataframe for the raw excel sheet and a dataframe
    # to be used in the aggregation.
    df_raw, df, site, site_code = _data_reader(stage_file)
    df_raw.drop('Unnamed: 0', axis=1)

    # Seperate data into years and creates an excel file for each year.
    start_year = df.index[0].to_pydatetime()
    start_year = start_year.year
    years = np.arange(start_year, 2017)
    years_str = np.array([str(x) for x in years])
    for year in years_str:
        df_year = df.loc[year]
        df_15_reindex, df_30_reindex, df_1h_reindex, df_1d_reindex = _resampler(
            df_year, site, year)

        # Saving data to a excel file if to_excel is True.
        if to_excel is True:
            if excel_filepath is None:
                save_path = (os.path.expanduser('~') + '/' + site + '_'
                             + str(year) + '_stage_data.xlsx')
            else:
                save_path = (excel_filepath + site + '_'
                             + str(year) + '_stage_data.xlsx')

            # Takes raw and each time interval of data and creates a sheet
            # for each.
            writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
            df_raw.to_excel(writer, 'raw_stationID' + site_code, index=False)
            df_15_reindex.to_excel(writer, '15min', index=False, na_rep='#N/A')
            df_30_reindex.to_excel(writer, '30min', index=False, na_rep='#N/A')
            df_1h_reindex.to_excel(writer, 'hourly', index=False, na_rep='#N/A')
            df_1d_reindex.to_excel(writer, 'daily', index=False, na_rep='#N/A')

            # Formatting of the excel sheets. Without format1 the time is saved
            # in decimal form in the excel sheet.
            workbook = writer.book
            format1 = workbook.add_format({'num_format': 'hh:mm:ss'})
            worksheet_raw = writer.sheets['raw_stationID' + site_code]
            worksheet_15 = writer.sheets['15min']
            worksheet_30 = writer.sheets['30min']
            worksheet_1h = writer.sheets['hourly']
            worksheet_1d = writer.sheets['daily']
            worksheets = [worksheet_15, worksheet_30, worksheet_1h, worksheet_1d]
            for worksheet in worksheets:
                worksheet.set_column('A:L', 20)
                worksheet.set_column('D:E', 20, format1)
            worksheet_raw.set_column('A:F', 20)
            writer.save()
            workbook.close()

        # Saving data to a csv file if to_csv is True.
        if to_csv is True:
            if csv_filepath is None:
                save_path_15 = (os.path.expanduser('~') + '/' + site + '_'
                                + str(year) + '_15min_stage_data.csv')
                save_path_30 = (os.path.expanduser('~') + '/' + site + '_'
                                + str(year) + '_30min_stage_data.csv')
                save_path_1h = (os.path.expanduser('~') + '/' + site + '_'
                                + str(year) + '_hourly_stage_data.csv')
                save_path_1d = (os.path.expanduser('~') + '/' + site + '_'
                                + str(year) + '_daily_stage_data.csv')
            else:
                save_path_15 = csv_filepath + site + '_15min_stage_data.csv'
                save_path_30 = csv_filepath + site + '_30min_stage_data.csv'
                save_path_1h = csv_filepath + site + '_hourly_stage_data.csv'
                save_path_1d = csv_filepath + site + '_daily_stage_data.csv'

            df_15_reindex.to_csv(save_path_15)
            df_30_reindex.to_csv(save_path_30)
            df_1h_reindex.to_csv(save_path_1h)
            df_1d_reindex.to_csv(save_path_1d)

        # Deletes dataframes after each year loop to save memory.
        del df_year, df_15_reindex, df_30_reindex, df_1h_reindex, df_1d_reindex
    # Deletes dataframes after the year loop is completed to save memory.
    del df_raw, df
    return


def _data_reader(file):
    """ Reads in raw data and creates variables needed for later
    saving the data. """
    # Create a dictionary so that filename matches a site name.
    site_dict = {'D05536000': 'NB Niles', 'D05536101': 'NS Channel-Wilmette',
                 'D05536105': 'NB Albany', 'D05536118': 'NB Grand Avenue',
                 'D05536121': 'CH River-Lock', 'D05536123': 'CH River-Columbus',
                 'D05536137': 'CSSC-Western Avenue', 'D05536140': 'CSSC-Stickney',
                 'D05536275': 'Thorn Creek', 'D05536290': 'Little Calument',
                 'D05536340': 'Midlothian Creek', 'D05536343': 'Natalie Creek',
                 'D05536357': 'Grand Calumet', 'D05536500': 'Tinley Creek',
                 'D05536700': 'Calumet-Sag Channel', 'D05536890': 'CSSC-Lemont',
                 'D05536995': 'CSSC-Romeoville'}
    df_raw = pd.read_csv(file)
    # Creating a dataframe with the data we only need.
    df = df_raw[['dateTime', 'X_00065_00000']]
    df = df.set_index(df_raw['dateTime'])

    # Retrieve site information to be used in saved excel filenames.
    site_code = file[-9:]
    site_name = [v for v in site_dict.items() if site_code in v][0]
    site_name[1].replace(' ', '-')
    site = site_code + '_' + site_name[1]

    # Convert index into a datetime index for easier indexing.
    df.index = pd.to_datetime(df.index)
    return df_raw, df, site, site_code


def _resampler(df_year, year):
    """ Takes raw data and aggregates it by mean and reindex so data
    begins in January and ends in December. """
    # Aggregates data using mean for each time interval.
    df_15 = df_year.resample('15T').mean()
    df_30 = df_year.resample('30T').mean()
    df_1h = df_year.resample('1H').mean()
    df_1d = df_year.resample('D').mean()

    # Creating new date range to include all time intervals within the year.
    idx_15 = pd.date_range(str(year) + '-01-01 00:15:00',
                           str(year) + '-12-31 23:45:00', freq='15T')
    idx_30 = pd.date_range(str(year) + '-01-01 00:30:00',
                           str(year) + '-12-31 23:30:00', freq='30T')
    idx_1h = pd.date_range(str(year) + '-01-01 00:00:00',
                           str(year) + '-12-31 23:00:00', freq='1H')
    idx_1d = pd.date_range(str(year) + '-01-01 00:00:00',
                           str(year) + '-12-31 23:00:00', freq='D')

    # Reindexing so data that starts in, for example August, will now
    # have the months prior to August filled with nans.
    df_15_reindex = df_15.reindex(idx_15, fill_value=np.nan)
    # Adding all columns to match example excel.
    df_15_reindex = df_15_reindex.rename(columns={'X_00065_00000': 'H(ft)'})
    # Adding meters column by dividing the feet column by 3.28.
    df_15_reindex['H(m)'] = df_15_reindex['H(ft)'] / 3.28
    df_15_reindex['DateTime2'] = df_15_reindex.index
    df_15_reindex['Date'] = df_15_reindex.index
    df_15_reindex['Date2'] = df_15_reindex.index
    df_15_reindex['Date_Python_generated'] = df_15_reindex['Date'].dt.date
    df_15_reindex['Time1'] = df_15_reindex['Date'].dt.time
    df_15_reindex['Time2'] = df_15_reindex['Date'].dt.time
    df_15_reindex['H(m)_final'] = df_15_reindex['H(m)']
    df_15_reindex = df_15_reindex.reset_index(drop=True)
    # Adding original datetime and height data to dataframe. To do this
    # pd.concat is used because the column lengths are different.
    df_15_reindex = pd.concat([
        df_15_reindex, df_year.reset_index(drop=True)], axis=1)
    # Reordering columns to match example excel.
    df_15_reindex = df_15_reindex[[
        'dateTime', 'X_00065_00000', 'Date_Python_generated', 'Time1', 'Time2',
        'DateTime2', 'Date', 'H(ft)', 'H(m)', 'Date2', 'H(m)_final']]
    # Filling nans with empty cells in columns similar to example excel.
    df_15_reindex[[
        'dateTime', 'X_00065_00000', 'H(m)_final'
    ]] = df_15_reindex[['dateTime', 'X_00065_00000', 'H(m)_final']].fillna('')

    # Similar to 15 minute interval code but 30 minutes interval.
    df_30_reindex = df_30.reindex(idx_30, fill_value=np.nan)
    df_30_reindex = df_30_reindex.rename(columns={'X_00065_00000': 'H(ft)'})
    df_30_reindex['H(m)'] = df_30_reindex['H(ft)'] / 3.28
    df_30_reindex['DateTime2'] = df_30_reindex.index
    df_30_reindex['Date'] = df_30_reindex.index
    df_30_reindex['Date2'] = df_30_reindex.index
    df_30_reindex['Date_Python_generated'] = df_30_reindex['Date'].dt.date
    df_30_reindex['Time1'] = df_30_reindex['Date'].dt.time
    df_30_reindex['Time2'] = df_30_reindex['Date'].dt.time
    df_30_reindex['H(m)_final'] = df_30_reindex['H(m)']
    df_30_reindex = df_30_reindex.reset_index(drop=True)
    df_30_reindex = pd.concat([
        df_30_reindex, df_year.reset_index(drop=True)], axis=1)
    df_30_reindex = df_30_reindex[[
        'dateTime', 'X_00065_00000', 'Date_Python_generated', 'Time1', 'Time2',
        'DateTime2', 'Date', 'H(ft)', 'H(m)', 'Date2', 'H(m)_final']]
    df_30_reindex[[
        'dateTime', 'X_00065_00000', 'H(m)_final'
    ]] = df_30_reindex[['dateTime', 'X_00065_00000', 'H(m)_final']].fillna('')

    # Similar to 15 minute interval code but hourly interval.
    df_1h_reindex = df_1h.reindex(idx_1h, fill_value=np.nan)
    df_1h_reindex = df_1h_reindex.rename(columns={'X_00065_00000': 'H(ft)'})
    df_1h_reindex['H(m)'] = df_1h_reindex['H(ft)'] / 3.28
    df_1h_reindex['DateTime2'] = df_1h_reindex.index
    df_1h_reindex['Date'] = df_1h_reindex.index
    df_1h_reindex['Date2'] = df_1h_reindex.index
    df_1h_reindex['Date_Python_generated'] = df_1h_reindex['Date'].dt.date
    df_1h_reindex['Time1'] = df_1h_reindex['Date'].dt.time
    df_1h_reindex['Time2'] = df_1h_reindex['Date'].dt.time
    df_1h_reindex['H(m)_final'] = df_1h_reindex['H(m)']
    df_1h_reindex = df_1h_reindex.reset_index(drop=True)
    df_1h_reindex = pd.concat([
        df_1h_reindex, df_year.reset_index(drop=True)], axis=1)
    df_1h_reindex = df_1h_reindex[[
        'dateTime', 'X_00065_00000', 'Date_Python_generated', 'Time1', 'Time2',
        'DateTime2', 'Date', 'H(ft)', 'H(m)', 'Date2', 'H(m)_final']]
    df_1h_reindex[[
        'dateTime', 'X_00065_00000', 'H(m)_final'
    ]] = df_1h_reindex[['dateTime', 'X_00065_00000', 'H(m)_final']].fillna('')

    # Similar to 15 minute interval code but daily interval.
    df_1d_reindex = df_1d.reindex(idx_1d, fill_value=np.nan)
    df_1d_reindex = df_1d_reindex.rename(columns={'X_00065_00000': 'H(ft)'})
    df_1d_reindex['H(m)'] = df_1d_reindex['H(ft)'] / 3.28
    df_1d_reindex['DateTime2'] = df_1d_reindex.index
    df_1d_reindex['Date'] = df_1d_reindex.index
    df_1d_reindex['Date2'] = df_1d_reindex.index
    df_1d_reindex['Date_Python_generated'] = df_1d_reindex['Date'].dt.date
    df_1d_reindex['Time1'] = df_1d_reindex['Date'].dt.time
    df_1d_reindex['Time2'] = df_1d_reindex['Date'].dt.time
    df_1d_reindex['H(m)_final'] = df_1d_reindex['H(m)']
    df_1d_reindex = df_1d_reindex.reset_index(drop=True)
    df_1d_reindex = pd.concat([
        df_1d_reindex, df_year.reset_index(drop=True)], axis=1)
    df_1d_reindex = df_1d_reindex[[
        'dateTime', 'X_00065_00000', 'Date_Python_generated', 'Time1', 'Time2',
        'DateTime2', 'Date', 'H(ft)', 'H(m)', 'Date2', 'H(m)_final']]
    df_1d_reindex[[
        'dateTime', 'X_00065_00000', 'H(m)_final'
    ]] = df_1d_reindex[['dateTime', 'X_00065_00000', 'H(m)_final']].fillna('')
    return df_15_reindex, df_30_reindex, df_1h_reindex, df_1d_reindex
