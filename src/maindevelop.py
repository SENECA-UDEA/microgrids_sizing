from utilities import read_data, calculate_cost_data
from utilities import read_multiyear_data, calculate_multiyear_data
from classes import RandomCreate
from mainfunctions import maindispatch, maindispatchmy, mainopt, mainopttstage 
from mainfunctions import mainstoc, mainstocmy
import copy
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

#folder path
folder_path = os.getcwd()

 
# file paths local
demand_filepath = "../data/" + PLACE + "/demand_" + PLACE+".csv"
forecast_filepath = "../data/"+PLACE+"/forecast_" + PLACE + ".csv"
units_filepath = "../data/" + PLACE + "/parameters_" + PLACE + ".json"
instanceData_filepath = "../data/" + PLACE + "/instance_data_" + PLACE + ".json"

#fiscal Data
fiscalData_filepath = "../data/auxiliar/fiscal_incentive.json"

#cost Data
costData_filepath = "../data/auxiliar/parameters_cost.json"



#select what model is going to be used
type_model = 'Stochastic'
#type_model = ['Deterministic','Stochastic','Multiyear','Stochastic-multiyear',
#'Optimization1stage','Optimization2stage']

#Different way to read model according to the model selected

if (type_model != 'Multiyear' and type_model != 'Stochastic-multiyear'):
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
    
    #run the model according to the type_model
    if (type_model == 'Deterministic'):
        percent_df, energy_df, renew_df, total_df, brand_df  = maindispatch(demand_df,
                                                                            forecast_df, generators, batteries, instance_data, fisc_data, cost_data,
                                                                            best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)
    elif (type_model == 'Stochastic' ):
        demand_df_i = copy.deepcopy(demand_df)
        forecast_df_i = copy.deepcopy(forecast_df)
        percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions = mainstoc(demand_df_i, 
                                                                                                              forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                                              best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)


        
    elif (type_model == 'Optimization1stage' or type_model == 'Optimization2stage'):
        #Set GAP
        #User selects gap and solver
        solver_data = {"MIP_GAP":0.01,"TEE_SOLVER":True,"OPT_SOLVER":"gurobi"}
        
        if (type_model == 'Optimization1stage'):
            percent_df, energy_df, renew_df, total_df, brand_df  = mainopt(demand_df,
                                                                           forecast_df, generators, batteries, 
                                                                           instance_data, fisc_data, cost_data, solver_data, folder_path)
        else:
            percent_df, energy_df, renew_df, total_df, brand_df, df_iterations = mainopttstage (demand_df, 
                                                                                                forecast_df, generators, batteries, instance_data, fisc_data,
                                                                                                cost_data, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, solver_data, folder_path)
    
else:
    #Multiyear
    #multiyear Data
    myearData_filepath = "../data/auxiliar/multiyear.json"
    
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
    
    #Calculate salvage, operation and replacement cost with investment cost
    generators, batteries = calculate_cost_data(generators, batteries, 
                                                instance_data, cost_data)


    if (type_model == 'Stochastic-multiyear'):
        demand_df_year = copy.deepcopy(demand_df_i)
        forecast_df_year = copy.deepcopy(forecast_df_i)
        
        percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions = mainstocmy(demand_df_year, 
                                                                                                                forecast_df_year, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                                                my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)
            
    else:
        demand_df, forecast_df = calculate_multiyear_data(demand_df_i, forecast_df_i,
                                                          my_data, instance_data['years'])  


        percent_df, energy_df, renew_df, total_df, brand_df  =  maindispatchmy(demand_df,
                                                                               forecast_df, generators, batteries, instance_data, fisc_data, cost_data,
                                                                               my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)

