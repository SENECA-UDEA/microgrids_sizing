# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

from utilities import read_data, create_objects, calculate_sizingcost, create_technologies, script_generators 
import opt as opt
import pandas as pd 
import random as random
from plotly.offline import plot
import copy
from classes import Solution
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


#Create data with n for solar and Wind
if (instance_data['create_generators'] == 'True'):
    generators_def = script_generators(generators, instance_data['amax'])
else:
    generators_def = generators



# Create objects and generation rule
generators_dict, batteries_dict = create_objects(generators_def,
                                                 batteries, 
                                                 forecast_df,
                                                 demand_df,
                                                 instance_data)

#Create technologies and renewables set
technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)



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
                       mintec = instance_data['mintec'], 
                       maxbr = instance_data['max_brand'],
                       years = instance_data['years'],
                       w_cost = instance_data['w_cost'],
                       tlpsp = instance_data['tlpsp'])    


print("Model generated")
# solve model 
results, termination = opt.solve_model(model, 
                       optimizer = 'gurobi',
                       mipgap = 0.01,
                       tee = True)
print("Model optimized")


#TODO:  ext_time?
if termination['Temination Condition'] == 'optimal': 
   model_results = opt.Results(model)
   print(model_results.descriptive)
   print(model_results.df_results)
   generation_graph = model_results.generation_graph()
   plot(generation_graph)
   column_data = {}
   for bat in batteries_dict.values(): 
       if (model_results.descriptive['batteries'][bat.id_bat] == 1):
           column_data[bat.id_bat+'_%'] =  model_results.df_results[bat.id_bat+'_b-'] / model_results.df_results['demand']
   for gen in generators_dict.values():
       if (model_results.descriptive['generators'][gen.id_gen] == 1):
           column_data[gen.id_gen+'_%'] =  model_results.df_results[gen.id_gen] / model_results.df_results['demand']
   
   percent_df = pd.DataFrame(column_data, columns=[*column_data.keys()])
