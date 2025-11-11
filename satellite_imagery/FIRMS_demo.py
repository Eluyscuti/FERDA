'''
Get an API Key from https://firms.modaps.eosdis.nasa.gov/api/map_key.
Put the following line in key.py:
    FIRMS_API_KEY = '<your-key-here>'

'''
import key # holds the FIRMS API key
import pandas as pd
import requests


# check how many transactions left in our key
url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + key.FIRMS_API_KEY
try:
    response = requests.get(url)
    data = response.json()
    df = pd.Series(data)
    print(df)
    #print('try in the browser: ', url)
except:
    print('error: try in the browser: ', url)
    


