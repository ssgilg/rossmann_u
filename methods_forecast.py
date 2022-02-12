# here all methods regarding forecast

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

from prophet import Prophet

def show_parameters(store,item, frequency):
    print("Store:     ", store)
    print("Item:      ", item)
    print('Frequency: ', frequency)


def prepare_data(data, Store, item, period):
    item_ = item
    item_string = 'item'+str(item_) 
    stores_type_ = data[data['Store']== Store]
    stores_type_ = stores_type_.groupby('Date').sum()
    stores_type_ = stores_type_[['Sales', item_string]]
    items_store_ = stores_type_.resample(period).agg({'Sales':np.sum, item_string: np.sum})
    items_store_['Date'] = items_store_.index
    ts = items_store_.rename(columns = {'Date': 'ds',  item_string : 'y'})
    ts = ts.fillna(0)
    return ts


def perform_forecast(data, f, future_days ):
    """
    data:        time series with date as index and columns y and ds
    f:           (int) frequency of time series
    future days: (int) How many days in the future you want to forecast
                 if a period of 3 months is needed , then future_days = 90
    """

    my_model = Prophet(interval_width = 0.75)
    my_model.fit(data)

    #  Create start and end time for future dates
    s , l =  time_interval_future_dates(data, future_days)
    future_dates =  pd.DataFrame({'ds': pd.date_range(s, l, freq=f)})
    
    forecast = my_model.predict(future_dates)
    
    print("Finished calculating Forecast " )
    
    return forecast, s, l


def time_interval_future_dates(some_df, prediction_period):
    
    
    delta = timedelta(days=prediction_period)
    
    last = some_df.ds.values[len(some_df)-1]
    last = pd.to_datetime(last)
    last = last + delta
    
    start = some_df.ds.values[0]
    start = pd.to_datetime(start)
    
    return start, last


def time_interval(some_df):
    one_year = timedelta(days=365)
    last = len(some_df)

    last = some_df.ds.values[last-1]
    last = pd.to_datetime(last)

    start = last - one_year 
    
    return start, last