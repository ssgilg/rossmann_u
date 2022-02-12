 # here : methods for unit_sales_bench.ipynb

import pandas as pd
import numpy as np    

from scipy.stats import norm

from datetime import datetime
from datetime import timedelta

    
    
    
def select_type_store(data, type_store, item_, year):
        item_string = 'item'+str(item_) 
        stores_type_ = data[data['StoreType']== type_store]
        stores_type_ = stores_type_.groupby('Date').sum()
        stores_type_ = stores_type_[stores_type_.index.year==year][['Sales', item_string]]
        items_store_ = stores_type_.resample('W-Fri').agg({'Sales':np.sum, item_string: np.sum})
        return  items_store_ 
    
    

    
def optimal_rev_period(t):
    lower_time = t / np.sqrt(2) 
    upper_time = t * np.sqrt(2)

    powers = []
    for m in range(0, 10):
        testing = (2**m)*1
        if lower_time <=  testing and  testing <= upper_time:
            powers.append(m)
        else:
            pass

    min_power = min(powers)
    R = 2**min_power * 1
    return R


def optimal_review_parameters(data, item, price, L, k, a):
    time = len(data)
    base = 52
    mean = np.mean(data[item])
    yearly_demand = np.sum(data[item])
    d_std = np.std(data[item])
    z = norm.ppf(a) 
    
    h_ = np.round(price * 0.1, 2)      #: yearly cost
    T_ = np.sqrt((2 * k) /( h_ * yearly_demand)) * base
    #T = np.ceil(np.sqrt((2 * k) /( h * yearly_demand)))
    
    #m = 2
    R = optimal_rev_period(T_)
    x_std = np.sqrt(L+R)*d_std
   
    return time, mean, yearly_demand, z, T_, R, x_std, h_


def get_up_to_level(x_std,z, mean, R, L):
    
    Ss = np.round(x_std*z).astype(int) 
    Cs = 1/2 * mean * R 
    Is = mean * L 
    S = Ss + 2*Cs + Is
    
    return Ss, Cs, Is, S


def create_arrays(data, time, L ):
    hand_ = np.zeros(time, dtype=int) 
    transit = np.zeros((time,L+1), dtype=int)
    stock_out_period = np.full(time, False, dtype=bool)
    stock_out_cycle = []
    stock_out_record = [0]*len(hand_)
    order = [0]*len(hand_)
   
    return hand_, transit, stock_out_period, stock_out_cycle, stock_out_record, order


def get_orders(df, hand, transit,stock_out_period, stock_out_cycle,stock_out_record, order, item, S, time,R_review, L):

    def get_data_item(data,item):
        d = data[item].values
        dates = data.index
        return d, dates
    
    d, dates = get_data_item(df, 'itemA')    


    hand[0] = S - d[0]
    transit[1,-1] = d[0]
    stock_out_record[0] = 'No Action' 


    for t in range(1,time):
        if transit[t-1,0]>0: 
            stock_out_cycle.append(stock_out_period[t-1]) 
        
        hand[t] = hand[t-1] - d[t] + transit[t-1,0] 
        stock_out_period[t] = hand[t] < 0
        transit[t,: -1] = transit[t -1,1:]
        if 0==t%R_review: 
            net = hand[t] + transit[t].sum() 
            transit[t,L] = S - net
            order[t] = S - net
        
        if hand[t] < d[t]:
            stock_out_record[t] = 'True'
        else:
            stock_out_record[t]= 'False'
        
    frame = pd.DataFrame(data= {'Demand':d, 'On-hand':hand, 'In−transit':list(transit), 'stock-out': stock_out_record, 'Order zise':order},
                         index =  df.index) 
    frame['OrderSize'] = frame['In−transit'].apply(lambda x: x[-1])
    
    return frame, stock_out_cycle, stock_out_period