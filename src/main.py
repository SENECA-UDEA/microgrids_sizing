# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

from utilities import read_data, create_objects, calculate_size
import opt as opt
import pandas as pd 

# file paths github
demand_filepath = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/Demanda%20Anual%20Leticia.csv' 
forecast_filepath = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/Forecast%20anual.csv' 
units_filepath  = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/parameters_P.json' 
# file paths local
#demand_filepath = "../data/Demanda Anual Leticia.csv"
#forecast_filepath = '../data/Forecast anual.csv'
#units_filepath = "../data/parameters_P.json"
demand_filepath = "../data/demand_day.csv"
forecast_filepath = '../data/forecast_day.csv'
units_filepath = "../data/parameters_P.json"


# read data
demand_df, forecast_df, generators, batteries = read_data(demand_filepath,
                                                          forecast_filepath,
                                                          units_filepath)
# Create objects
generators_dict, batteries_dict, technologies_dict, renewables_dict = create_objects(generators,
                                                                                   batteries)
# TODO: this is not needed
size = calculate_size(demand_df, forecast_df, generators_dict)


# Create model          
model = opt.make_model(generators_dict, 
                       forecast_df, 
                       batteries_dict, 
                       dict(zip(demand_df.t, demand_df.demand)),
                       technologies_dict, 
                       renewables_dict, 
                       20, 0.2, 0.8,4,3,2,20, size)    

# solve model 
results, termination = opt.solve_model(model, 
                       optimizer = 'gurobi',
                       mipgap = 0.02,
                       tee = True)
# TODO: check how are the termination conditions saved
# TODO: I thinkthe results can be gattered directly from results and not from model
if termination['Temination Condition'] == 'optimal': #check if the word is optimal or another
   model_results = opt.Results(model)
   print(model_results.descriptive)
   print(model_results.df_results)
      

      