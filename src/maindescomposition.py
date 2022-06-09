# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

from utilities import read_data, create_objects, calculate_sizingcost, create_technologies
import opt as opt
import pandas as pd 
import random as random
from operators import Sol_constructor, Search_operator
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

# Create objects and generation rule
generators_dict, batteries_dict,  = create_objects(generators,
                                                   batteries,  
                                                   forecast_df,
                                                   demand_df,
                                                   instance_data)

technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)


#create the initial solution operator
search_operator = Sol_constructor(generators_dict, 
                            batteries_dict,
                            demand_df,
                            forecast_df)

#create a default solution
sol_feasible = Sol_constructor.initial_solution(instance_data,
                                               generators_dict, 
                                               batteries_dict, 
                                               technologies_dict, 
                                               renewables_dict)

# set the initial solution as the best so far
sol_best = copy.deepcopy(sol_feasible)

# create the actual solution with the initial soluion
sol_current = copy.deepcopy(sol_feasible)

#check the available area
amax =  instance_data['amax']
movement = "Initial Solution"

#df of solutions
rows_df = []

# Create search operator
search_operator = Search_operator(generators_dict, 
                            batteries_dict,
                            demand_df,
                            forecast_df)

for i in range(20):
    rows_df.append([i, sol_current.feasible, 
                    sol_current.results.descriptive['area'], 
                    sol_current.results.descriptive['LCOE'], 
                    sol_best.results.descriptive['LCOE'], movement])
    if sol_current.feasible == True:     
        # save copy as the last solution feasible seen
        sol_feasible = copy.deepcopy(sol_current) 
        # Remove a generator or battery from the current solution
        sol_try = search_operator.removeobject(sol_current)
        movement = "Remove"
    else:
        #  Create list of generators that could be added
        list_available = search_operator.available(sol_current, amax)
        if list_available != []:
            # Add a generator or battery to the current solution
            #sol_try = search_operator.addobject(sol_current, list_available, demand_df)
            sol_try = search_operator.addrandomobject(sol_current, list_available)
            movement = "Add"
        else:
            # return to the last feasible solution
            sol_try = copy.deepcopy(sol_feasible)
            continue # Skip running the model and go to the begining of the for loop
    tnpc_calc, crf_calc = calculate_sizingcost(sol_try.generators_dict_sol, 
                                               sol_try.batteries_dict_sol, 
                                               ir = instance_data['ir'],
                                               years = instance_data['years'])
    model = opt.make_model_operational(generators_dict = sol_try.generators_dict_sol,
                                       batteries_dict = sol_try.batteries_dict_sol,  
                                       demand_df=dict(zip(demand_df.t, demand_df.demand)), 
                                       technologies_dict = sol_try.technologies_dict_sol,  
                                       renewables_dict = sol_try.renewables_dict_sol,
                                       nse =  instance_data['nse'], 
                                       TNPC = tnpc_calc,
                                       CRF = crf_calc,
                                       w_cost = instance_data['w_cost'],
                                       tlpsp = instance_data['tlpsp']) 
    
    results, termination = opt.solve_model(model, 
                                           optimizer = 'gurobi',
                                           mipgap = 0.02,
                                            tee = True)
    
    

    if termination['Temination Condition'] == 'optimal':
        sol_try.results.descriptive['LCOE'] = model.LCOE_value.expr()
        sol_try.results = opt.Results(model)
        sol_try.feasible = True
        sol_current = copy.deepcopy(sol_try)
        if sol_try.results.descriptive['LCOE'] <= sol_best.results.descriptive['LCOE']:
            sol_best = copy.deepcopy(sol_try)
    else:
        sol_try.feasible = False
        sol_try.results.descriptive['LCOE'] = None
        sol_current = copy.deepcopy(sol_try)

    sol_current.results.descriptive['area'] = search_operator.calculate_area(sol_current)
         
               
                
#df with the feasible solutions
df_iterations = pd.DataFrame(rows_df, columns=["i", "feasible", "area", "LCOE_actual", "LCOE_Best","Movement"])

column_data = {}
for bat in sol_best.batteries_dict_sol.values(): 
    if (sol_best.results.descriptive['batteries'][bat.id_bat] == 1):
        column_data[bat.id_bat+'_%'] =  sol_best.results.df_results[bat.id_bat+'_b-'] / sol_best.results.df_results['demand']
for gen in sol_best.generators_dict_sol.values(): 
    if (sol_best.results.descriptive['generators'][gen.id_gen] == 1):
        column_data[gen.id_gen+'_%'] =  sol_best.results.df_results[gen.id_gen] / sol_best.results.df_results['demand']
   
percent_df = pd.DataFrame(column_data, columns=[*column_data.keys()])


'''
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
