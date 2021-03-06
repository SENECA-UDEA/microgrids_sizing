# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

from utilities import read_data, create_objects
import opt as opt
import pandas as pd 
import random as random
from plotly.offline import plot
pd.options.display.max_columns = None

# file paths github
demand_filepath = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/Leticia_Annual_Demand.csv' 
forecast_filepath = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/Annual_Forecast.csv' 
units_filepath  = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/parameters_P.json' 
# file paths local
#demand_filepath = "../data/Leticia_Annual_Demand.csv"
#forecast_filepath = '../data/Annual_Forescast.csv'
#units_filepath = "../data/parameters_P.json"
demand_filepath = "../data/demand_day.csv"
forecast_filepath = '../data/forecast_day.csv'
units_filepath = "../data/parameters_P.json"
instanceData_filepath = "../data/instance_data.json"


# read data
demand_df, forecast_df, generators, batteries, instance_data = read_data(demand_filepath,
                                                          forecast_filepath,
                                                          units_filepath,
                                                          instanceData_filepath)

# Create objects and generation rule
generators_dict, batteries_dict, technologies_dict, renewables_dict = create_objects(generators,
                                                                                   batteries, forecast_df)


# Create model          
model = opt.make_model(generators_dict, 
                       batteries_dict, 
                       dict(zip(demand_df.t, demand_df.demand)),
                       technologies_dict, 
                       renewables_dict, 
                       amax = instance_data['amax'], 
                       ir = instance_data['ir'], 
                       nse = instance_data['nse'], 
                       maxtec = instance_data['maxtec'], 
                       maxbr = instance_data['max_brand'],
                       years = instance_data['years'],
                       tlpsp = instance_data['tlpsp'])    


print("Model generated")
# solve model 
results, termination = opt.solve_model(model, 
                       optimizer = 'gurobi',
                       mipgap = 0.01,
                       tee = True)
print("Model optimised")

# TODO: check how are the termination conditions saved
#TODO:  ext_time?
if termination['Temination Condition'] == 'optimal': 
   model_results = opt.Results(model)
   print(model_results.descriptive)
   print(model_results.df_results)
   generation_graph = model_results.generation_graph()
   plot(generation_graph)
   
'''
# Run model decomposition
 
n_gen = 6
generators = random.sample(generators, n_gen)
n_bat = 1
batteries = random.sample(batteries, n_bat)
# Create objects and generation rule
generators_dict, batteries_dict, technologies_dict, renewables_dict = create_objects(generators,
                                                                                   batteries,  forecast_df)

model = opt.make_model_operational(generators_dict=generators_dict, 
                               batteries_dict=batteries_dict,  
                               demand_df=dict(zip(demand_df.t, demand_df.demand)), 
                               technologies_dict = technologies_dict,  
                               renewables_dict = renewables_dict,
                               nse =  instance_data['nse'], 
                               TNPC = instance_data['TNPC'],
                               CRF = instance_data['CRF'],
                               tlpsp = instance_data['tlpsp'])      
# solve model 
results, termination = opt.solve_model(model, 
                       optimizer = 'gurobi',
                       mipgap = 0.02,
                       tee = True)
if termination['Temination Condition'] == 'optimal': 
   model_results = opt.Results(model)
   print(model_results.descriptive)
   print(model_results.df_results)
   generation_graph = model_results.generation_graph()
   plot(generation_graph)
'''
