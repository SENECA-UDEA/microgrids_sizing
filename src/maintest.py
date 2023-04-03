from utilities import read_data, calculate_cost_data
from utilities import read_multiyear_data, calculate_multiyear_data
from classes import RandomCreate
from mainfunctions import maindispatch, maindispatchmy, mainopt, mainopttstage 
from mainfunctions import mainstoc, mainstocmy
import os

#Set the seed for random

#SEED = 42

SEED = None

rand_ob = RandomCreate(seed = SEED)

#add and remove funtion
ADD_FUNCTION = 'GRASP'
REMOVE_FUNCTION = 'RANDOM'

#data PLACE
PLACE = 'Providencia'

#time not served best solution
best_nsh = 0

#Set GAP
'''User selects gap and solver'''
solver_data = {"MIP_GAP":0.01,"TEE_SOLVER":True,"OPT_SOLVER":"gurobi"}

 
# file paths local
demand_filepath = "../data/" + PLACE + "/demand_" + PLACE+".csv"
forecast_filepath = "../data/"+PLACE+"/forecast_" + PLACE + ".csv"
units_filepath = "../data/" + PLACE + "/parameters_" + PLACE + ".json"
instanceData_filepath = "../data/" + PLACE + "/instance_data_" + PLACE + ".json"

#fiscal Data
fiscalData_filepath = "../data/auxiliar/fiscal_incentive.json"

#cost Data
costData_filepath = "../data/auxiliar/parameters_cost.json"

#multiyear Data
myearData_filepath = "../data/auxiliar/multiyear.json"


# read data
demand_df, forecast_df, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                forecast_filepath,
                                                                                                units_filepath,
                                                                                                instanceData_filepath,
                                                                                                fiscalData_filepath,
                                                                                                costData_filepath)


#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, 
                                            instance_data, cost_data)
#Demand to be covered
demand_df['demand'] = instance_data['demand_covered'] * demand_df['demand'] 


'''MULTIYEAR'''

# read my data
demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data, my_data = read_multiyear_data(demand_filepath,
                                                                                                                      forecast_filepath,
                                                                                                                      units_filepath,
                                                                                                                      instanceData_filepath,
                                                                                                                      fiscalData_filepath,
                                                                                                                      costData_filepath,
                                                                                                                      myearData_filepath)


#Demand to be covered
demand_df['demand'] = instance_data['demand_covered'] * demand_df['demand'] 

demand_df, forecast_df = calculate_multiyear_data(demand_df_i, forecast_df_i,
                                                  my_data, instance_data['years'])        


#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, 
                                            instance_data, cost_data)

'''STOCHASTIC'''
# read data
demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                forecast_filepath,
                                                                                                units_filepath,
                                                                                                instanceData_filepath,
                                                                                                fiscalData_filepath,
                                                                                                costData_filepath)


#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, 
                                            instance_data, cost_data)
#Demand to be covered
demand_df_i['demand'] = instance_data['demand_covered'] * demand_df_i['demand'] 


'''STOCHASTIC MY'''
# read data
demand_df_year, forecast_df_year, generators, batteries, instance_data, fisc_data, cost_data, my_data = read_multiyear_data(demand_filepath,
                                                                                                                      forecast_filepath,
                                                                                                                      units_filepath,
                                                                                                                      instanceData_filepath,
                                                                                                                      fiscalData_filepath,
                                                                                                                      costData_filepath,
                                                                                                                      myearData_filepath)

#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, 
                                            instance_data, cost_data)
#Demand to be covered
demand_df_year['demand'] = instance_data['demand_covered'] * demand_df_year['demand']

folder_path = os.getcwd()


percent_df, energy_df, renew_df, total_df, brand_df  = maindispatch(demand_df,
                                                                    forecast_df, generators, batteries, instance_data, fisc_data, cost_data,
                                                                    best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)

percent_df, energy_df, renew_df, total_df, brand_df  =  maindispatchmy(demand_df, forecast_df, generators, batteries, instance_data, fisc_data, cost_data,
                                                                       my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)

percent_df, energy_df, renew_df, total_df, brand_df  = mainopt(demand_df,forecast_df, generators, batteries, 
                                                               instance_data, fisc_data, cost_data, solver_data, folder_path)

percent_df, energy_df, renew_df, total_df, brand_df, df_iterations = mainopttstage (demand_df, forecast_df, generators, batteries, instance_data, fisc_data,
                                                                                    cost_data, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, solver_data, folder_path)


percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions = mainstoc(demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                                      best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)


percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions = mainstocmy(demand_df_year, forecast_df_year, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                                        my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)