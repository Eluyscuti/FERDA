'''
Get an API Key from https://firms.modaps.eosdis.nasa.gov/api/map_key.
Put the following line in key.py:
    FIRMS_API_KEY = '<your-key-here>'

You get 5000 credits per 10 minutes, which is more than enough.

'''
import key # holds the FIRMS API key
import pandas as pd
import requests

debug = False
show_credits_used = True

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
    


if __name__ == '__main__':
    start_transactions = check_key_transactions()

    #date_df = check_date_range('all')
    date_df = check_date_range('GOES_NRT')
    print(date_df)





    end_transactions = check_key_transactions()
    if show_credits_used:
        print(f'Used {end_transactions - start_transactions} credits this run. {5000 - end_transactions} credits remain.')