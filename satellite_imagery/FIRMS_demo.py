'''
Get an API Key from https://firms.modaps.eosdis.nasa.gov/api/map_key.
Put the following line in key.py:
    FIRMS_API_KEY = '<your-key-here>'

You get 5000 credits per 10 minutes, which is more than enough.

'''
import key # holds the FIRMS API key
import pandas as pd
import geopandas as gpd
import plotnine
import requests

debug = False
show_credits_used = True

# keep states_df as a global so we don't have to reread it a lot
states_df = None

# check how many transactions left in our key
def check_key_transactions():
    '''
    Args:
    Returns:
        current_transactions (int): number of transactions used currently

    '''
    url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + key.FIRMS_API_KEY
    current_transactions = -1
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.Series(data)
        if debug:
            print(df)
        current_transactions = df['current_transactions']
        #print('try in the browser: ', url)
    except:
        print('error: try in the browser: ', url)
    return current_transactions

def check_date_range(data_id):
    '''
    prints the date range of available data from FIRMS satellites. Uses up to 5 transactions.

    Args:
        data_id (string): Options for data_id are listed below:

        (the ones of most interest to us have an arrow)
        (NRT is near real time, while SP is standard product (more latency but more accurate), BA = burned area)

        MODIS_NRT           <--
        MODIS_SP
        VIIRS_NOAA20_NRT    <-- 
        VIIRS_NOAA20_SP
        VIIRS_NOAA21_NRT    <--
        VIIRS_SNPP_NRT
        VIIRS_SNPP_SP
        LANDSAT_NRT
        GOES_NRT            <-- 
        BA_VIIRS
        all                 <-- get info of all datasets
    
    Returns:
        pandas dataframe

    '''
    url = 'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/' + key.FIRMS_API_KEY + '/' + data_id
    try:
        df = pd.read_csv(url)
        #print(df)
        return df
    except:
        print(f"error in getting data availability of '{data_id}'. Please make sure it is a valid option.")
    
def query_fire_detection(source, area_coordinates='world', date=1):
    # in this example let's look at VIIRS NOAA-20, entire world and the most recent day
    #url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + key.FIRMS_API_KEY + '/VIIRS_NOAA20_NRT/world/1'
    # focusing on just asia, last three days of records
    # area_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_NOAA20_NRT/54,5.5,102,40/3'
    url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + key.FIRMS_API_KEY + '/' + source + '/' + area_coordinates + '/' + str(date)
    try:
        df = pd.read_csv(url)
        return df
    except:
        pass

def display_fire_data(df):

    # Map abbreviations to full words
    conf_map = {'l': 'low', 'n': 'nominal', 'h': 'high'}
    df = df.copy()
    df['confidence'] = df['confidence'].map(conf_map)

    plot = (
        plotnine.ggplot(df)
        + plotnine.geom_point(plotnine.aes(x='longitude', y='latitude', color='confidence'))
        + plotnine.labs(title=f"Fire Data from {df['instrument'].iloc[0]}", color='Confidence Level')
        + plotnine.theme(figure_size=(14, 6))
    )

    plot.show()

    # TODO: 
    # overlay US map onto it.
    # adjust df to have only california or united states.


if __name__ == '__main__':
    start_transactions = check_key_transactions()

    satellite = 'VIIRS_NOAA20_NRT'

    # show date ranges for GOES
    date_df = check_date_range(satellite)# do date_df = check_date_range('all') to get date ranges for all satellites 
    print(date_df)

    # try to get one day of fire detection (from VIIRS NRT, of the whole world, past 1 day)
    area_df = query_fire_detection('VIIRS_NOAA20_NRT', 'world', 1)
    print(area_df.columns)
    display_fire_data(area_df)

    end_transactions = check_key_transactions()
    if show_credits_used:
        if end_transactions >= 0:
            print(f'Used {end_transactions - start_transactions} credits this run. {5000 - end_transactions} credits remain.')
        else:
            print('error in check_key_transactions()')