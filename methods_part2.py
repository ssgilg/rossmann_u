# Here : methods for rossmann_part2_inventory

import pandas as pd
import numpy as np
import math

def choose_data(df, n_store, item, year, frequency):
    
    store_id     = df.query("Store == " + str(n_store))
    data_ts_     = store_id[['Sales', item]]
    store_id     = store_id[store_id.index.year==year][['Sales', item]]
    agg_store_id = store_id.resample(frequency).sum()
    agg_store_series = pd.Series(data_ts_.resample('W').sum().itemA)
    return agg_store_series, agg_store_id



def average_demand_item_X(data, item, S, price, lead_time, time_unit):
    y_demand = np.round(data[item].sum(),2)
    
    D = np.round(data[item].mean(),2)
    H = np.round(price * 0.1, 2)
    C = np.round((price - H) * 0.6, 2)
   
    
    # Economical Order Quantity
    Q = np.ceil(np.sqrt((2 * y_demand* S) / H))
    
    # Reorder point. In items units
    R = np.floor(D*lead_time)
    
    # Total inventory cost
    tc = D * C + (D * S) / Q + (Q * H)/2
    TC = np.round(tc, 2)
    
    # Optimal interval to place orders in time_units
    T = np.round((Q/D) * time_unit, 1)
    return item, D, Q, R, TC, T, y_demand, H



def round_up_EQO(Q):

    q = math.ceil(Q/100)
    q = q*100
    return q



def optimal_replenishment_time(fixed_transaction_cost: float, 
                                   yearly_demand: float, 
                                   holding_cost: float):
    D = yearly_demand
    h = holding_cost # Get yearly holding cost
    k = fixed_transaction_cost # Get transaction cost
    
    # Get the optimal review time. This is given as percentage of a year and Match it with the base time
    t = np.sqrt((2 * k)/ (h * D)) * 183
    
    # Get the review period in power of two
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
    optimal_replenishment_time = 2**min_power * 1
     
    return optimal_replenishment_time



def get_orders(data_, item,store, i_inventory,R, lead_time, eqo, fixed_cost, y_demand, h, safe: bool = True):
    
    df = data_[item]
    time = len(df)
    on_hand = np.zeros(time,dtype=int)
    transit = np.zeros((time, lead_time+1), dtype=int )
    
    stockout_period = np.full(time, False, dtype=bool)
    stockout_cycle  = []
    
    init_inventory = i_inventory  ## inventory at intial time
    #stock = init_inventory + eqo 
    if safe ==True:
        stock = init_inventory + eqo
    else:
        stock = init_inventory
    
    on_hand[0] = stock - df.iloc[0]
    transit[0,-1] = eqo 
    #stock_list = [0]*time
    #stock_list[0] = stock
    #stock_out_record = [0]*time

    for record in  range(1, time):
        if transit[record-1,0] > 0:
            stockout_cycle.append(stockout_period[record-1])         
            
        on_hand[record] = on_hand[record -1 ] - df.iloc[record] + transit[record -1,0]
        if  on_hand[record] <= R:
             transit[record,lead_time] = eqo
                
        stockout_period[record] = on_hand[record]<0
        transit[record, :-1] = transit[record-1,1:]
                
        if 0==record%lead_time:
            net = on_hand[record] + transit[record].sum()
            
    SL_alpha  = 1-sum(stockout_cycle)/len(stockout_cycle)
    SL_period = 1-sum(stockout_period)/time
    
    frame = pd.DataFrame(data = { 'itemA':df, 'On-hand':on_hand, 'In-transit': list(transit)})
    #frame['stockout']  = stock_out_period
    frame['OrderSize'] = frame['In-transit'].apply(lambda x: x[-1])        
    #frame['Incoming']  = frame['In-transit'].apply(lambda x: x[0]) 
#         stockout_period[t] = on_hand[t] < 0
#         transit[t,:-1] = transit[t-1,1:]
#         diff = stock_list[record] - data_item[record]
#         if diff <= R:
#             diff = diff +  EQO*num_orders
#             notes_ = 'Order'
#         else:
#             notes_ = 'No Order'
#         notes.append(notes_)
#         stock_list[record + 1] = diff
#         #pint(data_item[record],  stock_list[record], notes_) 
#     frame = pd.DataFrame({'Date':data_item.index, 'itemA': data_item.values, 'Action': notes})
    return frame, stockout_cycle, stockout_period


def stockouts(df, array):
    no_stockouts = array.sum()
    num_orders = df[df['OrderSize']>0]['OrderSize'].count()
    
    print("Stockouts :", no_stockouts)
    print("num orders :", num_orders)
    
    
    
def SL_alpha(stockout_cycle):
    sl = 1-sum(stockout_cycle)/len(stockout_cycle)
    return round(sl*100,1) 
    
def SL_period(stockout_period, periods):
    sl = 1-sum(stockout_period)/periods
    return round(sl*100,1)  



def order_dates(df):

    order_dates = []
    for index, row in df.iterrows():
        if row['OrderSize']> 0:
            order_dates.append(index)
            
    return order_dates


def cost_per_order(fixed_transaction_cost, y_demand, holding_cost, order_quantity):
    holding_cost = (holding_cost * order_quantity)/2
    transaction_costs =  (fixed_transaction_cost * y_demand) / order_quantity
    cost_per_order = holding_cost + transaction_costs
    return cost_per_order

def get_model_costs(df, fix_cost,y_demand, h_cost):
    k = fix_cost
    D = y_demand 
    h_ = h_cost
    f = lambda x: 0 if x == 0 else cost_per_order(k,D,h_, x)
    total_cost = df['OrderSize'].apply(f).sum()
    return total_cost