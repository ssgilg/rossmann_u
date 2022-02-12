# here create db for transactions

import numpy as np
import pandas as pd
from pandas import datetime

from random import randint
# importing train data to learn
train = pd.read_csv("data/train.csv", 
                    parse_dates = True, low_memory = False, index_col = 'Date')


# first glance at the train set: head and tail
print("In total: ", train.shape)
train.head(5).append(train.tail(5))



cash_     =  68
giro_     =  26
transfer_ =   3
other     =   3



def make_tuples():
    
    def cash():
        return  randint(64, 72)
    

    def giro():
        return randint( 24 , 28 )


    def transfer():
        return randint(  2, 4)
    
    
    cash = cash()
    giro = giro()
    transfer = transfer()
    t =  cash + giro + transfer
    #print(cash, giro, transfer,t)
    
    if t > 100 :
        make_tuples()
    else:
        other = 100 - t 
        if other < 0:
            other = 0
        
    other = 100 - t
    if other < 0:
        other = 0
    return cash, giro, transfer, other
    
    

#print(type(make_tuples()))
#train['cash'] = cash.Customers.apply(lambda x: cash() )
# train['giro'] = cash.Customers.apply(lambda x: giro() )
# train['transfer'] = cash.Customers.apply(lambda x: cash() )

list_cash = []
list_giro = []
list_transfer = []
list_other = []

for row in train.iterrows():
    c, t, ct, o = make_tuples()
    if o < 0:
        print(c,t,ct,o)
        
    list_cash.append(c)
    list_giro.append(t)
    list_transfer.append(ct)
    list_other.append(o)
    

train['cash']     = list_cash * train['Customers']
train['giro']     = list_giro * train['Customers']
train['transfer'] = list_transfer * train['Customers']
train['other']    = list_other * train['Customers']

train['cash'] = train['cash'].apply(lambda x: x * 0.01)
train['giro'] = train['giro'].apply(lambda x: x * 0.01)
train['transfer'] = train['transfer'].apply(lambda x: x * 0.01)
train['other'] = train['other'].apply(lambda x: x * 0.01)
#print(train)
    
train.to_csv('train_transactions.csv', encoding = 'utf-8')