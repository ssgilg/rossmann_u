# here plot methods and outliers methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change shape of columns


def choose_col(df,name):
    df_XX =  df[name].dropna()
    X = np.array(df_XX)
    X = X.reshape(len(X),1) 
    
    X_f = X #X_f = X[1:int(len(X)/5)]
    return X_f

def quartil(X):
    data = pd.DataFrame(X)
    q1 = data.quantile(.25)               # Calculate quantiles 
    q2 = data.quantile(.5)
    q3 = data.quantile(.75)
    dif = q3 - q1
    k = 1.5
    
    Total = len(data)
    
    min, max = q1 - k*dif, q3 + k*dif     # Caculate interval
    min = min[0]
    max = max[0]
    print("Quartilte method, k = ", k)
    print("Min: ", min)
    print("Max: ",max)
    num_out = 0
    num_in = 0
    filtered = []
    
    for i in X:

        if (i[0].item() > max or  i[0].item() < min ):
            num_out = num_out + 1

        else:
            filtered.append(i[0])
            num_in = num_in + 1
    
    total = num_in + num_out
    Fraction = num_out/len(X)
    R_frac = num_out/Total
   
    print("fraction of outliers in a a total of " , total, " is: ", Fraction) 
    print("fraction of outliers in a a total of " , Total, " is: ", R_frac)
    
    return filtered, num_out, Fraction, R_frac


def plot_distr(df,name_col, method):
    X_f = choose_col(df, name_col)
    min_ = min(X_f)[0]
    max_ = max(X_f)[0]
    r, _ , _, _= method(X_f)

    plt.figure(figsize=(12,22))

    #Plot 1
    plt.subplot(4,2,1)
    #Axes.set_ylim(min_, max_)
    plt.plot(X_f)
    plt.ylim(min_, max_)
    plt.title('Line plot original data')
    # plt.xlabel('Week')
    plt.tick_params(labelbottom = False, bottom = False)

    #Plot 2
    plt.subplot(4,2,2)
    plt.hist(X_f, bins = 100)
    plt.title('Distribution of original data')
    #plt.tick_params(left = False, right = False , labelleft = False ,
    #                labelbottom = False, bottom = False)

    # LOF Detection for outliers
    plt.subplot(4,2,3)
    
    plt.plot(r, color = 'orange')
    plt.ylim(min_, max_)
    plt.title("Line plot with  outliers filtered")
    #plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)

    plt.subplot(4,2,4)
    #plt.hist(r, bins= 100, color = 'orange', range =[min_, max_])
    plt.hist(r, bins= 100, color = 'orange')
    plt.title("Distribution with outliers filtered")
    #plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    plt.show()