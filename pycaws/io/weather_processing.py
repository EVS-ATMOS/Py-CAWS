""" Code that takes raw weather data and does mean and standard
deviation calculations. Processed data can then be saved as excel. """

import os

import numpy as np
import pandas as pd


def weather_output(filename, to_excel=True, excel_filepath=None):
    """
    Weather output, takes in raw weather data and does preprocessing.

    Parameter
    ---------
    filename : str
        Filename of csv with all the weather station data.

    Optional Parameters
    -------------------
    to_excel : bool
        Choose whether or not to write the data to a excel file. Default is True.
    excel_filepath : str
        Filepath location to save excel file to. Default is None and the excel file
        is saved to user's home directory.

    """

    site_dict = {'WBAN:04807': 'Gary', 'WBAN:04831': 'Romeoville',
                 'WBAN:04838': 'Palwaukee', 'WBAN:04879': 'Lansing',
                 'WBAN:14819': 'Midway', 'WBAN:94846': 'Ohare'}

    df = pd.read_csv(
        filename, index_col='STATION',
        usecols=['STATION', 'STATION_NAME', 'DATE', 'HOURLYDRYBULBTEMPC'])

    stations = ['WBAN:04807', 'WBAN:04831', 'WBAN:04838',
                'WBAN:04879', 'WBAN:14819', 'WBAN:94846']

    for station in stations:
        df_raw = df.loc[station]
        df_raw.insert(loc=0, column='STATION', value=df_raw.index)
        df_raw = df_raw.reset_index(drop=True)

        df_sta = _station_roll_stats(df, station)
        df_sta.insert(loc=0, column='Date', value=df_sta.index)
        df_sta = df_sta.reset_index(drop=True)

        site_match = [v for v in site_dict.items() if station in v][0]
        site = site_match[1]
        # Saving data to a excel file if to_excel is True.
        if to_excel is True:
            if excel_filepath is None:
                save_path = (os.path.expanduser('~') + '/'
                             + site + '_preprocessed_weather_data.xlsx')
            else:
                save_path = (excel_filepath
                             + site + '_preprocessed_weather_data.xlsx')

            writer = pd.ExcelWriter(save_path, engine='xlsxwriter',
                                    datetime_format='m/d/yyyy h:mm')

            df_raw.to_excel(writer, 'raw_' + site, index=False)
            df_sta.to_excel(writer, 'AirTemp_C', index=False)

            # Formatting of the excel sheets. Without format1 the time is
            # saved in decimal form in the excel sheet.
            workbook = writer.book
            worksheet_raw = writer.sheets['raw_' + site]
            worksheet_sta = writer.sheets['AirTemp_C']

            worksheet_raw.set_column('A:D', 22)
            worksheet_raw.set_column('B:B', 24)
            worksheet_sta.set_column('A:U', 22)
            worksheet_sta.set_column('B:B', 24)
            writer.save()
            workbook.close()
        del df_raw
        del df_sta
    del df


def _station_roll_stats(df, station):
    """ Does the rolling stats on AirTemp_C for differing time periods. """
    df_sta = df.loc[station]
    df_sta = df_sta.set_index('DATE')
    df_sta.index = pd.to_datetime(df_sta.index)
    df_sta = df_sta.rename(columns={'HOURLYDRYBULBTEMPC': 'AirTemp_C'})

    df_sta['AirTemp_C'] = df_sta['AirTemp_C'].apply(_replace_invalid)
    df_sta['AirTemp_C'] = df_sta['AirTemp_C'].apply(
        pd.to_numeric, errors='coerce')

    # To ignore NaNs, we need to explicitly calculate mean and standard deviation
    df_sta['Tmean1'] = (df_sta['AirTemp_C'].rolling('1H', min_periods=1).sum()
                        / df_sta['AirTemp_C'].rolling('1H', min_periods=1).count())
    df_sta['Tmean2'] = (df_sta['AirTemp_C'].rolling('2H', min_periods=2).sum()
                        / df_sta['AirTemp_C'].rolling('2H', min_periods=2).count())
    df_sta['Tmean6'] = (df_sta['AirTemp_C'].rolling('6H', min_periods=6).sum()
                        / df_sta['AirTemp_C'].rolling('6H', min_periods=6).count())
    df_sta['Tmean12'] = (df_sta['AirTemp_C'].rolling('12H', min_periods=12).sum()
                         / df_sta['AirTemp_C'].rolling('12H', min_periods=12).count())
    df_sta['Tmean24'] = (df_sta['AirTemp_C'].rolling('24H', min_periods=24).sum()
                         / df_sta['AirTemp_C'].rolling('24H', min_periods=24).count())
    df_sta['Tmean48'] = (df_sta['AirTemp_C'].rolling('48H', min_periods=48).sum()
                         / df_sta['AirTemp_C'].rolling('48H', min_periods=48).count())
    df_sta['Tmean72'] = (df_sta['AirTemp_C'].rolling('72H', min_periods=72).sum()
                         / df_sta['AirTemp_C'].rolling('72H', min_periods=72).count())
    df_sta['Tmean96'] = (df_sta['AirTemp_C'].rolling('96H', min_periods=96).sum()
                         / df_sta['AirTemp_C'].rolling('96H', min_periods=96).count())
    df_sta['Tmean120'] = (df_sta['AirTemp_C'].rolling('120H', min_periods=120).sum()
                          / df_sta['AirTemp_C'].rolling('120H', min_periods=120).count())

    # Standard deviation
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean1']).apply(
        np.square).rolling('1H', min_periods=1).sum()
    df_sta['Tstd1'] = (temp/(df_sta['AirTemp_C'].rolling(
        '1H', min_periods=1).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean2']).apply(
        np.square).rolling('2H', min_periods=2).sum()
    df_sta['Tstd2'] = (temp/(df_sta['AirTemp_C'].rolling(
        '2H', min_periods=2).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean6']).apply(
        np.square).rolling('6H', min_periods=6).sum()
    df_sta['Tstd6'] = (temp/(df_sta['AirTemp_C'].rolling(
        '6H', min_periods=6).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean12']).apply(
        np.square).rolling('12H', min_periods=12).sum()
    df_sta['Tstd12'] = (temp/(df_sta['AirTemp_C'].rolling(
        '12H', min_periods=12).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean24']).apply(
        np.square).rolling('24H', min_periods=24).sum()
    df_sta['Tstd24'] = (temp/(df_sta['AirTemp_C'].rolling(
        '24H', min_periods=24).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean48']).apply(
        np.square).rolling('48H', min_periods=48).sum()
    df_sta['Tstd48'] = (temp/(df_sta['AirTemp_C'].rolling(
        '48H', min_periods=48).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean72']).apply(
        np.square).rolling('72H', min_periods=72).sum()
    df_sta['Tstd72'] = (temp/(df_sta['AirTemp_C'].rolling(
        '72H', min_periods=72).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean96']).apply(
        np.square).rolling('96H', min_periods=96).sum()
    df_sta['Tstd96'] = (temp/(df_sta['AirTemp_C'].rolling(
        '96H', min_periods=96).count()-1)).apply(np.sqrt)
    temp = df_sta['AirTemp_C'].subtract(df_sta['Tmean120']).apply(
        np.square).rolling('120H', min_periods=120).sum()
    df_sta['Tstd120'] = (temp/(df_sta['AirTemp_C'].rolling(
        '120H', min_periods=120).count()-1)).apply(np.sqrt)

    # To ignore NaNs, we need to explicitly calculate mean and standard deviation
    df_sta['Tsum1'] = df_sta['AirTemp_C'].rolling('1H', min_periods=1).sum()
    df_sta['Tsum2'] = df_sta['AirTemp_C'].rolling('2H', min_periods=2).sum()
    df_sta['Tsum6'] = df_sta['AirTemp_C'].rolling('6H', min_periods=6).sum()
    df_sta['Tsum12'] = df_sta['AirTemp_C'].rolling('12H', min_periods=12).sum()
    df_sta['Tsum24'] = df_sta['AirTemp_C'].rolling('24H', min_periods=24).sum()
    df_sta['Tsum48'] = df_sta['AirTemp_C'].rolling('48H', min_periods=48).sum()
    df_sta['Tsum72'] = df_sta['AirTemp_C'].rolling('72H', min_periods=72).sum()
    df_sta['Tsum96'] = df_sta['AirTemp_C'].rolling('96H', min_periods=96).sum()
    df_sta['Tsum120'] = df_sta['AirTemp_C'].rolling(
         '120H', min_periods=120).sum()

    # To ignore NaNs, we need to explicitly calculate mean and standard deviation
    df_sta['Tmin1'] = df_sta['AirTemp_C'].rolling('1H', min_periods=1).max()
    df_sta['Tmin2'] = df_sta['AirTemp_C'].rolling('2H', min_periods=2).max()
    df_sta['Tmin6'] = df_sta['AirTemp_C'].rolling('6H', min_periods=6).max()
    df_sta['Tmin12'] = df_sta['AirTemp_C'].rolling('12H', min_periods=12).max()
    df_sta['Tmin24'] = df_sta['AirTemp_C'].rolling('24H', min_periods=24).max()
    df_sta['Tmin48'] = df_sta['AirTemp_C'].rolling('48H', min_periods=48).max()
    df_sta['Tmin72'] = df_sta['AirTemp_C'].rolling('72H', min_periods=72).max()
    df_sta['Tmin96'] = df_sta['AirTemp_C'].rolling('96H', min_periods=96).max()
    df_sta['Tmin120'] = df_sta['AirTemp_C'].rolling(
        '120H', min_periods=120).max()

    # To ignore NaNs, we need to explicitly calculate mean and standard deviation
    df_sta['Tmin1'] = df_sta['AirTemp_C'].rolling('1H', min_periods=1).min()
    df_sta['Tmin2'] = df_sta['AirTemp_C'].rolling('2H', min_periods=2).min()
    df_sta['Tmin6'] = df_sta['AirTemp_C'].rolling('6H', min_periods=6).min()
    df_sta['Tmin12'] = df_sta['AirTemp_C'].rolling('12H', min_periods=12).min()
    df_sta['Tmin24'] = df_sta['AirTemp_C'].rolling('24H', min_periods=24).min()
    df_sta['Tmin48'] = df_sta['AirTemp_C'].rolling('48H', min_periods=48).min()
    df_sta['Tmin72'] = df_sta['AirTemp_C'].rolling('72H', min_periods=72).min()
    df_sta['Tmin96'] = df_sta['AirTemp_C'].rolling('96H', min_periods=96).min()
    df_sta['Tmin120'] = df_sta['AirTemp_C'].rolling(
        '120H', min_periods=120).min()

    # Difference between first and last
    df_sta['Tdiff1'] = df_sta['AirTemp_C'].rolling(
        '1H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff2'] = df_sta['AirTemp_C'].rolling(
        '2H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff6'] = df_sta['AirTemp_C'].rolling(
        '6H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff12'] = df_sta['AirTemp_C'].rolling(
        '12H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff24'] = df_sta['AirTemp_C'].rolling(
        '24H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff48'] = df_sta['AirTemp_C'].rolling(
        '48H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff72'] = df_sta['AirTemp_C'].rolling(
        '72H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff96'] = df_sta['AirTemp_C'].rolling(
        '96H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])
    df_sta['Tdiff120'] = df_sta['AirTemp_C'].rolling(
        '120H', min_periods=1, closed='both').apply(lambda x: x[-1] - x[0])

    # Round all answers to 1 decimal place (original
    # precision of temperature data).
    df_sta = df_sta.round(1)
    df_sta.fillna('')
    return df_sta


def _replace_invalid(value):
    """ Converts invalid temperature data to nans. """
    if isinstance(value, str):
        if 's' or '*' in value:
            value = np.nan
    return float(value)
