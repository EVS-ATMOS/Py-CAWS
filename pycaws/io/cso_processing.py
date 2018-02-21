""" Module that reads combined sewer overflow data, does analysis,
and exports the data to an excel file. """

from datetime import datetime, timedelta
import itertools

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

    rows = [pd.DataFrame({'minute':[m], 'site': [row['site']], 'gpm': [row['gpm']]})
            for ind, row in ranges.iterrows()
            for m in minutes.loc[row.parsed_start:row.parsed_stop]]

    min_ranges = pd.concat(rows)
    min_ranges.set_index(['minute', 'site'], inplace=True)
    min_ranges = min_ranges.groupby(level=['minute', 'site']).mean()
    indy = pd.MultiIndex.from_tuples(
        list(itertools.product(minutes.values, sites)), names=['minute', 'site'])

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
            dt = row['parsed_start'] + timedelta(minutes=int(row['Duration..min.']))
    except KeyError:
        try:
            dt = datetime.strptime(str(row['Stop.Date…Time']), fmt)
        except ValueError:
            dt = row['parsed_start'] + timedelta(minutes=int(row['Duration..min']))
    return dt
