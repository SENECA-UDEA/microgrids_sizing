# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

from src.utilities import read_data, create_objects, create_technologies, calculate_energy, interest_rate
from src.utilities import fiscal_incentive, calculate_cost_data
import src.opt as opt
import pandas as pd 
from plotly.offline import plot
pd.options.display.max_columns = None


#Algortyhm data
#Set GAP
Solver_data = {"MIP_GAP":0.01,"TEE_SOLVER":True,"OPT_SOLVER":"gurobi"}



#Instance Data
place = 'Providencia'
place = 'Test'
TRM = 3910

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

# read data
demand_df, forecast_df, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                forecast_filepath,
                                                                                                units_filepath,
                                                                                                instanceData_filepath,
                                                                                                fiscalData_filepath,
                                                                                                costData_filepath)

#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, instance_data, cost_data)
# Create objects and generation rule
generators_dict, batteries_dict = create_objects(generators,
                                                 batteries, 
                                                 forecast_df,
                                                 demand_df,
                                                 instance_data)

#Create technologies and renewables set
technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)

#Demand to be covered
demand_df['demand'] = instance_data['demand_covered']  * demand_df['demand'] 

#Calculate interest rate
ir = interest_rate(instance_data['i_f'],instance_data['inf'])



#Calculate fiscal incentives
delta = fiscal_incentive(fisc_data['credit'], 
                         fisc_data['depreciation'],
                         fisc_data['corporate_tax'],
                         ir,
                         fisc_data['T1'],
                         fisc_data['T2'])

# Create model          
model = opt.make_model(generators_dict, 
                       batteries_dict, 
                       dict(zip(demand_df.t, demand_df.demand)),
                       technologies_dict, 
                       renewables_dict, 
                       amax = instance_data['amax'], 
                       fuel_cost =  instance_data['fuel_cost'],
                       ir = ir, 
                       nse = instance_data['nse'], 
                       years = instance_data['years'],
                       splus_cost = instance_data['splus_cost'],
                       sminus_cost = instance_data['sminus_cost'],
                       tlpsp = instance_data['tlpsp'],
                       delta = delta,
                       greed = instance_data['inverter_greed_cost'],
                       nse_cost = cost_data['NSE_COST'])    


print("Model generated")
# solve model 
results, termination = opt.solve_model(model, 
                                        Solver_data)
print("Model optimized")



if termination['Temination Condition'] == 'optimal': 
   model_results = opt.Results(model, generators_dict)
   print(model_results.descriptive)
   print(model_results.df_results)
   generation_graph = model_results.generation_graph(0,len(demand_df))
   plot(generation_graph)
   try:
       percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(batteries_dict, generators_dict, model_results, demand_df)
   except KeyError:
       pass

#Calculate LCOE in Colombia current - COP

LCOE_COP = TRM * model_results.descriptive['LCOE']

#Create Excel File
'''
percent_df.to_excel("percentresults.xlsx")
model_results.df_results.to_excel("results.xlsx") 
'''