# -*- coding: utf-8 -*-

from src.utilities import read_data

#import opt as opt
from src.classes import Random_create
import pandas as pd 

pd.options.display.max_columns = None

import math

import scipy.stats as st

import numpy as np

place = 'Providencia'


'''
place = 'San_Andres'
place = 'Puerto_Nar'
place = 'Leticia'
place = 'Test'
place = 'Oswaldo'
'''
github_rute = 'https://raw.githubusercontent.com/SENECA-UDEA/microgrids_sizing/development/data/'
# file paths github
demand_filepath = github_rute + place+'/demand_'+place+'.csv' 
forecast_filepath = github_rute + place+'/forecast_'+place+'.csv' 
units_filepath = github_rute + place+'/parameters_'+place+'.json' 
instanceData_filepath = github_rute + place+'/instance_data_'+place+'.json' 
fiscalData_filepath = github_rute +'fiscal_incentive.json'
 
# file paths local
demand_filepath = "../data/"+place+"/demand_"+place+".csv"
forecast_filepath = "../data/"+place+"/forecast_"+place+".csv"
units_filepath = "../data/"+place+"/parameters_"+place+".json"
instanceData_filepath = "../data/"+place+"/instance_data_"+place+".json"

#fiscal Data
fiscalData_filepath = "../data/Cost/fiscal_incentive.json"

#cost Data
costData_filepath = "../data/Cost/parameters_cost.json"

#Set the seed for random
'''
seed = 42
'''
seed = None

rand_ob = Random_create(seed = seed)

# read data

demand_df, forecast_df, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                forecast_filepath,
                                                                                                units_filepath,
                                                                                                instanceData_filepath,
                                                                                                fiscalData_filepath,
                                                                                                costData_filepath)


taxx = 0.02
taxx2 = 0.03
taxx4 = 0.04
n_years = 20
len_total = 8760*n_years


dd = {k : [0]*(len_total) for k in demand_df}
dd2 = {k : [0]*(len_total) for k in forecast_df}

for i in range(len_total):
    dd['t'][i] = i
    dd2['t'][i] = i
    if (i < 8760):    
        dd['demand'][i]= demand_df['demand'][i]
        dd2['DNI'][i] = forecast_df['DNI'][i]
        dd2['t_ambt'][i] = forecast_df['t_ambt'][i]
        dd2['Wt'][i] = forecast_df['Wt'][i]
        dd2['Qt'][i] = forecast_df['Qt'][i]
        dd2['GHI'][i] = forecast_df['GHI'][i]
        dd2['day'][i] = forecast_df['day'][i]
        dd2['SF'][i] = forecast_df['SF'][i]
        dd2['DHI'][i] = forecast_df['DHI'][i]
    else:
        k = math.floor(i/8760)
        val = demand_df['demand'][i - 8760*k] * (1 + taxx)**k
        dd['demand'][i] = val
        val2 = forecast_df['DNI'][i-8760*k]*(1 + taxx2)**k
        dd2['DNI'][i] = val2
        dd2['t_ambt'][i] = forecast_df['t_ambt'][i-8760*k]
        val3 = val2 = forecast_df['Wt'][i-8760*k]*(1 + taxx4)**k
        dd2['Wt'][i] = val3
        dd2['Qt'][i] = forecast_df['Qt'][i-8760*k]
        val4 = forecast_df['GHI'][i-8760*k]*(1 + taxx2)**k
        dd2['GHI'][i] = val4
        dd2['day'][i] = forecast_df['day'][i-8760*k]
        dd2['SF'][i] = forecast_df['SF'][i-8760*k]
        val5 = forecast_df['DHI'][i-8760*k]*(1 + taxx2)**k
        dd2['DHI'][i] = val5      
        
        
demand = pd.DataFrame(dd, columns=['t','demand'])
forecast = pd.DataFrame(dd2, columns=['t', 'DNI', 't_ambt', 'Wt', 'Qt', 'GHI', 'day', 'SF', 'DHI'])

fuel_cost = 1
tax = 0.01

len_total = 8760*n_years
for i in range(len_total):
    k = math.floor(i/8760)
    fuel_cost_r = fuel_cost * (1 + tax)**k
    #val = gen.DG_max * (1-tax)**k
    
#degradación de los equipos

def get_best_distribution(data):
    dist_names = ["exponweib","norm", "weibull_max", "weibull_min", "pareto", "genextreme", 
                  "gamma", "beta", "rayleigh", "invgauss","uniform","expon",   
                  "lognorm","pearson3","triang"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def get_best_distribution2(data):
    dist_names = ["exponweib","norm", "weibull_max", "weibull_min", "pareto", "genextreme", 
                  "gamma", "beta", "rayleigh", "invgauss","uniform","expon",   
                  "lognorm","pearson3"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(demand)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]
from scipy.stats import pearson3
skew, loc1, scale1 = pearson3.fit(demand)

#np.random.distribution(parameters)
nn = len(demand_df['t'])/24
dem_vec = {k : [0]*int(nn) for k in range(int(24))}
wind_vec = {k : [0]*int(nn) for k in range(int(24))}
sol_vec = {k : [0]*int(nn) for k in range(int(24))}

for t in demand_df['t']:
    l = t%24
    k = math.floor(t/24)
    dem_vec[l][k] = demand_df['demand'][t]

for k in forecast_df['t']:
    l = t%24
    k = math.floor(t/24)
    wind_vec[l][k] = forecast_df['Wt'][t]
    sol_vec[l][k] = forecast_df['DNI'][t]

a,b,c = get_best_distribution2(demand)
pp = [1,2,3,4,5,6,7,8,9,8,7,5,10,100,20,2,5,4]
a,b,c = get_best_distribution2(pp)

for k in dem_vec:
    a, b, c = get_best_distribution(dem_vec[k])

'normal'
s = np.random.normal(c[0], c[1], 1)
n = s[0]

'uniform'
s = np.random.uniform(c[0], c[1],1)
n = s[0]

'traingular'
s = np.random.triangular(c[0], c[1],c[2],2)
n = s[0]


'beta'
s = np.random.beta(a1,b1,1)
n = s[0]
