""" Module that reads combined sewer overflow data, does analysis,
and exports the data to an excel file. """

from datetime import datetime, timedelta
import glob
import itertools
import os
import shutil
import sys

import numpy as np
import pandas as pd


def cso_output(filename, site_name, excel_filepath):
    """
    Parameters
    ----------
    filename : str
        Filename of the excel document with the site's cso data.
    site_name : str
        The name of the site being processed. For example: 'OBrien'.
    excel_filepath : str
        Filepath location to save excel file to. This is a directory,
        actual filename is created based on site information.

    """
    main_output, df_15, df_1h, year = _cso_processing(filename)

    save_path = (excel_filepath + site_name
                 + '_' + str(year) + '_' + 'CSO_Summary.xlsx')

    writer = pd.ExcelWriter(save_path, engine='xlsxwriter',
                            datetime_format='m/d/yyyy h:mm')

    main_output.to_excel(writer, '1-minute_CSOs', index=True)
    df_15.to_excel(writer, '15-minute_mean', index=True)
    df_1h.to_excel(writer, 'hourly_mean', index=True)

    # Formatting of the excel sheet.
    workbook = writer.book
    worksheet_main = writer.sheets['1-minute_CSOs']
    worksheet_15 = writer.sheets['15-minute_mean']
    worksheet_1h = writer.sheets['hourly_mean']

    worksheet_main.set_column('A:A', 22)
    worksheet_main.set_column('B:AW', 16)
    worksheet_15.set_column('A:A', 22)
    worksheet_15.set_column('B:AW', 16)
    worksheet_1h.set_column('A:A', 22)
    worksheet_1h.set_column('B:AW', 16)

    writer.save()
    workbook.close()
    del main_output, df_15, df_1h


def cso_preparation(file, site_name, directory=None):
    """

    Parameters
    ----------
    file : str
        Name of a file to be processed.
    site_name : str
        Name of site to add to save name. Example: 'Calumet'

    Optional Parameters
    -------------------
    directory : str
        Location to create the folders for each site and year.
        Example: '/User/home/pycaws/data/'

    """
    # Read in the cso file.
    df_cso = pd.read_csv(file)

    # Turn index into DateTimeIndex.
    df_cso = df_cso.set_index('DATE')
    df_cso.index = pd.to_datetime(df_cso.index)

    # Create a folder pertaining to site and year.
    year = str(df_cso.index.year[0])
    if directory is None:
        # Check if os is Windows, because Windows handles
        # filepaths differently.
        if sys.platform == 'windows':
            directory = os.path.expanduser('~') + '\\'
        else:
            directory = os.path.expanduser('~') + '/'
    if sys.platform == 'windows':
        site_dir = directory + site_name + '_' + year + '\\'
    else:
        site_dir = directory + site_name + '_' + year + '/'
    if not os.path.exists(site_dir):
        os.makedirs(site_dir)

    # Convert gpm to m^3/s.
    df_cso = df_cso * 0.0000631

    # Take data from every 5 minutes and fill 0.0 when data isn't
    # at that interval.
    df_5 = df_cso.resample('5T').asfreq().fillna(0.0)
    max_df = pd.DataFrame({'CSO_name': df_5.columns, 'Qmax': 0.0})
    for column in df_5.columns:
        out_df = pd.DataFrame(df_5[column], index=df_5.index)
        out_df.index = out_df.index.strftime('%m/%d/%Y %H:%M')
        save_name = site_dir + column + '.csv'
        out_df.to_csv(save_name)

        max_df.loc[
            max_df['CSO_name'] == column, 'Qmax'] = out_df.values.max()
        # Deletes a DataFrame to save memory.
        del out_df
    max_df = max_df.reset_index(drop=True)
    max_df.to_csv(
        site_dir + 'MaxQ_Values_' + site_name + '_' + year + '.csv')
    # Deletes remaining DataFrames to save memory.
    del df_cso, df_5, max_df


def cso_aggregation(year, duflow_excel, directory=None):
    """
    Used to do aggregation of sites found in part B.

    Parameters
    ----------
    year : int
        The year for which data should be ran, 2014 or 2015.
    duflow_excel : str
        Location of the duflow excel sheet to be used.

    Optional Parameters
    -------------------
    directory : str
        Locations to save all folders and files for parts A and B.
        Default will save to the user's home directory.mro
    """
    if directory is None:
    # Check if os is Windows, because Windows handles
    # filepaths differently.
        if sys.platform == 'windows':
            directory = os.path.expanduser('~') + '\\'
        else:
            directory = os.path.expanduser('~') + '/'

    str_year = str(year)

    # Gathers files for all three sites and puts them into a
    # temporary directory.
    files_cal = glob.glob(directory + 'Calumet_' + str_year + '/*')
    files_sti = glob.glob(directory + 'Stickney_' + str_year + '/*')
    files_obr = glob.glob(directory + 'OBrien_' + str_year + '/*')

    temp_dir = directory + 'temp_dir/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for file in files_cal:
        shutil.copy(file, temp_dir)

    for file in files_sti:
        shutil.copy(file, temp_dir)

    for file in files_obr:
        shutil.copy(file, temp_dir)

    # Removes copys of MaxQ files, these exists in site year folders.
    maxqs = glob.glob(temp_dir + 'MaxQ*')
    for maxq in maxqs:
        os.remove(maxq)

    # Combines all duplicates.
    temp_files = glob.glob(temp_dir + '*')
    parsed = [(file, _parse_site_name(file)) for file in temp_files]

    # Checks to see if the CSO_Aggregation_year directory exists,
    # if not, makes the directory.
    dest_directory = directory + 'CSO_Aggregation_' + str_year + '/'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    sites = {}
    for raw, p in parsed:
        if p not in sites.keys():
            sites[p] = []
        sites[p].append(raw)

    for site, paths in sites.items():
        dup_dfs = [_prep_df(path) for path in paths]
        combined_df = dup_dfs[0]
        for other in dup_dfs[1:]:
            combined_df += other
        combined_df.to_csv(dest_directory + site + '.csv', index=True)

    # Changes name of DSM104 to DSM104E to match excel sheet.
    if os.path.exists(
            dest_directory + 'DSM104.csv'):
        os.rename(dest_directory + 'DSM104.csv',
                  dest_directory + 'DSM104E.csv')

    # Checks to see if the CSO_Aggregation_year_Final directory exists,
    # if not, makes the directory.
    final_directory = directory + 'CSO_Aggregation_' + str_year + '_Final/'
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # Reads in the duflow excel sheet, removes rows as explained
    # in instructions and changes nans to 0s in the columns because
    # pandas apply doesn't play well with nans.
    df_duflow = pd.read_excel(duflow_excel)
    drop_list = [12, 21, 22, 26, 36, 39, 40, 41, 42, 43,
                 45, 46, 47, 51, 52, 53, 57, 60, 62, 63]
    drop_list = [i-1 for i in drop_list]
    df_duflow = df_duflow.drop(df_duflow.index[[drop_list]])

    df_duflow = df_duflow.replace(np.nan, '0')
    df_duflow['Name and  description'] = df_duflow[
        'Name and  description'].astype(str)

    # Apply the _site_column_parser function to each row.
    df_duflow.apply(
        _site_column_parser,
        args=(str_year, dest_directory, final_directory),
        axis=1)

    # Removes the temporary directory.
    shutil.rmtree(temp_dir)

    # Deletes DataFrames to save memory.
    del combined_df, dup_dfs, df_duflow


def _cso_processing(file):
    """ Takes raw CSO data from an excel file and puts the date into the
    format found in the examples of csv. The times between start and stop
    time are matched with the times in the csv and those times are filled
    in with the gpm values for that time. 15 minute and hourly mean
    resampling is done. """
    df_raw = pd.read_excel(file)
    try:
        parsed_starts = pd.to_datetime(df_raw['Start.Date...Time'])
    except KeyError:
        parsed_starts = pd.to_datetime(df_raw['Start.Date…Time'])
    parsed_starts = pd.to_datetime(df_raw['Start.Date...Time'])
    df_raw['parsed_start'] = parsed_starts
    parsed_stops = df_raw.apply(_parse_dates, axis=1)
    df_raw['parsed_stop'] = parsed_stops
    year = parsed_starts[1].year
    idx_min = pd.date_range(str(year) + '-01-01 00:00:00',
                            str(year) + '-12-31 23:59:00', freq='1T')
    minutes = pd.Series([str(m) for m in idx_min], index=idx_min)

    sites = df_raw['DS..'].unique()
    ranges = df_raw[['parsed_start', 'parsed_stop']].copy()
    ranges['site'] = df_raw['DS..']
    ranges['gpm'] = df_raw['Intensity..gpm']
    ranges['idx'] = ranges.index.values

    rows = [pd.DataFrame({'minute':[m], 'site': [row['site']],
                          'gpm': [row['gpm']]})
            for ind, row in ranges.iterrows()
            for m in minutes.loc[row.parsed_start:row.parsed_stop]]

    min_ranges = pd.concat(rows)
    min_ranges.set_index(['minute', 'site'], inplace=True)
    min_ranges = min_ranges.groupby(level=['minute', 'site']).mean()
    indy = pd.MultiIndex.from_tuples(
        list(itertools.product(
            minutes.values, sites)), names=['minute', 'site'])

    output = pd.DataFrame(index=indy)
    output = output.join(min_ranges)
    main_output = output.reset_index().pivot(
        index='minute', columns='site', values='gpm')
    main_output = main_output.reset_index()
    main_output = main_output.rename(columns={'minute': 'DATE'})
    main_output.columns.name = None

    main_output.set_index('DATE')
    main_output.index = pd.to_datetime(main_output.DATE)
    main_output = main_output.drop(['DATE'], axis=1)

    df_15 = main_output.resample('15T').mean()
    df_1h = main_output.resample('1H').mean()

    del df_raw, ranges, min_ranges, rows, output
    return main_output, df_15, df_1h, year


def _parse_dates(row):
    """ If datetime is in float format, function changes format to
    that seen in the start datetime. """
    fmt = '%-m/%-d/%Y %-H:%-M'
    try:
        try:
            dt = datetime.strptime(str(row['Stop.Date...Time']), fmt)
        except ValueError:
            dt = row['parsed_start'] + timedelta(
                minutes=int(row['Duration..min.']))
    except KeyError:
        try:
            dt = datetime.strptime(str(row['Stop.Date…Time']), fmt)
        except ValueError:
            dt = row['parsed_start'] + timedelta(
                minutes=int(row['Duration..min']))
    return dt


def _site_column_parser(row, str_year, dest_directory, final_directory):
    """ A function that takes the sites found in columns 5 through 9 and
    adds the data together and creates the new file based on the column
    'Name and description'. In cases with no sites in columns 5 through 9,
    the data is set to 0, and in cases with 1 site, the data is set to
    that site. """
    # Creates a list of sites from columns 5 through 9.
    col_list = row[['CSO allocation', 'Unnamed: 5',
                    'Unnamed: 6', 'Unnamed: 7',
                    'Unnamed: 8']].values.tolist()
    col_list = np.array(col_list)

    # If a column has no site, the index is deleted from the list.
    new_list = np.delete(col_list, np.where(col_list == '0'))
    file_list = np.array(
        [dest_directory
         + i + '.csv' for i in new_list])
    file_list = [s.strip() for s in file_list]

    # If a site is found on the excel sheet, but not in the site data,
    # removes it from the filename list to avoid error. An example is
    # the 'TGI32' site appears in the excel sheet, but not in the 2014
    # data.
    real_list = []
    for file in file_list:
        if os.path.exists(file) is True:
            real_list += [file]

    # Creates a fresh DataFrame with 5 minute intervals.
    idx_min = pd.date_range(str_year + '-01-01 00:00:00',
                            str_year + '-12-31 23:55:00', freq='5T')
    site_df = pd.DataFrame({'DATE': idx_min, 'm3/s': 0.0})
    site_df = site_df.set_index('DATE')

    # Retrieves the name to be used in saving the final csv file.
    name = 'D' + row[['Name and  description']].values.tolist()[0]

    # If no sites appear in that row, sets 'm3/s' column to all 0.0s.
    if not real_list:
        site_df.to_csv(final_directory + name + '.csv', index=True)
    # If more then one site, combines that data and sets the column
    # 'm3/s' to that.
    else:
        dfs = [_prep_df(file) for file in real_list]
        agg = dfs[0]
        for other in dfs[1:]:
            agg += other
        agg.to_csv(final_directory + name + '.csv', index=True)


def _prep_df(path):
    """ Preps DataFrames before being placed in a list. """
    df = pd.read_csv(path)
    df.columns = ['DATE', 'm3/s']
    df.set_index('DATE', inplace=True)
    return df


def _parse_site_name(name):
    """ Seperates site from filename. Used to see if there
    are duplicates. """
    base = name.split('/')[-1]
    no_ext = base.split('.')[0]
    no_space = no_ext.split(' ')[0]
    return no_space
