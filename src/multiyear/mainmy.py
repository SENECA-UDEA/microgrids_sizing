# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

"""
from src.multiyear.utilitiesmy import read_data, create_technologies
from src.multiyear.utilitiesmy import calculate_area, calculate_energy
from src.multiyear.utilitiesmy import fiscal_incentive, calculate_cost_data
from src.multiyear.utilitiesmy import calculate_multiyear_data, interest_rate
from src.multiyear.utilitiesmy import calculate_inverter_cost, create_objects
from src.multiyear.classesmy import RandomCreate
import pandas as pd 
from src.multiyear.operatorsmy import SolConstructor, SearchOperator
from plotly.offline import plot
from src.multiyear.strategiesmy import select_strategy, ds_battery_renewable
from src.multiyear.strategiesmy import ds_diesel_renewable, ds_diesel 
from src.multiyear.strategiesmy import Results, ds_dies_batt_renew
import copy
pd.options.display.max_columns = None

#Set the seed for random

#SEED = 42

SEED = None

rand_ob = RandomCreate(seed = SEED)

#add and remove funtion
ADD_FUNCTION = 'GRASP'
REMOVE_FUNCTION = 'RANDOM'

#data PLACE
PLACE = 'Providencia'

'''
PLACE = 'San_Andres'
PLACE = 'Puerto_Nar'
PLACE = 'Leticia'
'''
#trm to current COP
TRM = 3910
#time not served best solution
best_nsh = 0

#Strategy list for select
list_ds_diesel = ["diesel"]
list_ds_diesel_renewable = [
    "diesel - solar","diesel - wind", 
    "diesel - solar - wind"
    ]

list_ds_battery_renewable = [
    "battery - solar","battery - wind",
    "battery - solar - wind"
    ]

list_ds_dies_batt_renew = [
    "battery - diesel - wind","battery - diesel - solar", 
    "battery - diesel - solar - wind"
    ]

loc_file = '/SENECA-UDEA/microgrids_sizing/development/data/'
github_rute = 'https://raw.githubusercontent.com' + loc_file

# file paths github
demand_filepath = github_rute + PLACE + '/demand_' + PLACE + '.csv' 
forecast_filepath = github_rute + PLACE + '/forecast_' + PLACE + '.csv' 
units_filepath = github_rute + PLACE + '/parameters_' + PLACE + '.json' 
instanceData_filepath = github_rute + PLACE + '/instance_data_' + PLACE + '.json' 
fiscalData_filepath = github_rute +'fiscal_incentive.json'
 
# file paths local
demand_filepath = "../../data/" + PLACE + "/demand_" + PLACE+".csv"
forecast_filepath = "../../data/"+PLACE+"/forecast_" + PLACE + ".csv"
units_filepath = "../../data/" + PLACE + "/parameters_" + PLACE + ".json"
instanceData_filepath = "../../data/" + PLACE + "/instance_data_" + PLACE + ".json"

#fiscal Data
fiscalData_filepath = "../../data/auxiliar/fiscal_incentive.json"

#cost Data
costData_filepath = "../../data/auxiliar/parameters_cost.json"

#multiyear Data
myearData_filepath = "../../data/auxiliar/multiyear.json"

# read data
demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data, my_data = read_data(demand_filepath,
                                                                                                        forecast_filepath,
                                                                                                        units_filepath,
                                                                                                        instanceData_filepath,
                                                                                                        fiscalData_filepath,
                                                                                                        costData_filepath,
                                                                                                        myearData_filepath)

#calculate multiyear data
demand_df, forecast_df = calculate_multiyear_data(demand_df_i, forecast_df_i,
                                                  my_data, instance_data['years'])

#calulate parameters
AMAX = instance_data['amax'] 
N_ITERATIONS = instance_data['N_iterations']
#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, 
                                            instance_data, cost_data)
#Demand to be covered
demand_df['demand'] = instance_data['demand_covered'] * demand_df['demand'] 

#Calculate interest rate
ir = interest_rate(instance_data['i_f'], instance_data['inf'])
 
#Calculate fiscal incentives
delta = fiscal_incentive(fisc_data['credit'], 
                         fisc_data['depreciation'],
                         fisc_data['corporate_tax'],
                         ir,
                         fisc_data['T1'],
                         fisc_data['T2'])

# Create objects and generation rule
generators_dict, batteries_dict = create_objects(generators,
                                                 batteries,  
                                                 forecast_df,
                                                 demand_df,
                                                 instance_data,
                                                 my_data)
#create technologies
technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)

#check diesel or batteries and at least one generator, for feasibility
if ('D' in technologies_dict.keys() or 'B' in technologies_dict.keys() 
    and generators_dict != {}):
    #create the initial solution operator
    sol_constructor = SolConstructor(generators_dict, 
                                     batteries_dict,
                                     demand_df,
                                     forecast_df)
    
    #create a default solution
    sol_feasible = sol_constructor.initial_solution(instance_data,
                                                   technologies_dict, 
                                                   renewables_dict,
                                                   delta,
                                                   rand_ob,
                                                   cost_data,
                                                   my_data,
                                                   ir)
    
    #if use aux_diesel asigns a big area to avoid select it again
    if ('aux_diesel' in sol_feasible.generators_dict_sol.keys()):
        generators_dict['aux_diesel'] = sol_feasible.generators_dict_sol['aux_diesel']
        generators_dict['aux_diesel'].area = 10000000
    
    #calculate area
    sol_feasible.results.descriptive['area'] = calculate_area(sol_feasible)    
    # set the initial solution as the best so far
    sol_best = copy.deepcopy(sol_feasible)
    
    # create the actual solution with the initial soluion
    sol_current = copy.deepcopy(sol_feasible)

    #inputs for the model
    movement = "Initial Solution"
    
    #df of solutions
    rows_df = []
    
    # Create search operator
    search_operator = SearchOperator(generators_dict, 
                                     batteries_dict,
                                     demand_df,
                                     forecast_df)
    
    #check that first solution is feasible
    if (sol_best.results != None):
        for i in range(N_ITERATIONS):
            '''
            ILS Procedure
            '''
            #create df to export results
            rows_df.append([i, sol_current.feasible, 
                            sol_current.results.descriptive['area'], 
                            sol_current.results.descriptive['LCOE'], 
                            sol_best.results.descriptive['LCOE'], movement])
            
            if sol_current.feasible:     
                # save copy as the last solution feasible seen
                sol_feasible = copy.deepcopy(sol_current) 
                # Remove a generator or battery from the current solution
                if (REMOVE_FUNCTION == 'GRASP'):
                    sol_try, remove_report = search_operator.remove_object(sol_current, delta)
                elif (REMOVE_FUNCTION == 'RANDOM'):
                    sol_try, remove_report = search_operator.remove_random_object(sol_current,
                                                                                  rand_ob)
    
                movement = "Remove"
            else:
                #  Create list of generators that could be added
                list_available_bat, list_available_gen, list_tec_gen = search_operator.available_items(sol_current
                                                                                                       , AMAX)
                
                if (list_available_gen != [] or list_available_bat != []):
                    # Add a generator or battery to the current solution
                    if (ADD_FUNCTION == 'GRASP'):
                        sol_try, remove_report = search_operator.add_object(sol_current, 
                                                                            list_available_bat, list_available_gen, list_tec_gen, 
                                                                            remove_report, instance_data['fuel_cost'], rand_ob, delta)
                    
                    elif (ADD_FUNCTION == 'RANDOM'):
                        sol_try = search_operator.add_random_object(sol_current, 
                                                                    list_available_bat, list_available_gen, list_tec_gen, rand_ob)
                    
                    movement = "Add"
                else:
                    # return to the last feasible solution
                    sol_current = copy.deepcopy(sol_feasible)
                    continue # Skip running the model and go to the begining of the for loop
    
            #defines which dispatch strategy to use
            strategy_def = select_strategy(generators_dict = sol_try.generators_dict_sol,
                                           batteries_dict = sol_try.batteries_dict_sol) 
            
            #calculate inverter cost with installed generators
            #val = instance_data['inverter_cost']#first of the functions
            #instance_data['inverter cost'] = calculate_inverter_cost(sol_try.generators_dict_sol,sol_try.batteries_dict_sol,val)
    
            print("defined strategy")
            #run the dispatch strategy
            if (strategy_def in list_ds_diesel):
                lcoe_cost, df_results, state, time_f, nsh = ds_diesel(sol_try, demand_df, 
                                                                      instance_data, cost_data, my_data, ir)
                
            elif (strategy_def in list_ds_diesel_renewable):
                lcoe_cost, df_results, state, time_f, nsh = ds_diesel_renewable(sol_try, 
                                                                                demand_df, instance_data, cost_data, delta, my_data, ir)
                
            elif (strategy_def in list_ds_battery_renewable):
                lcoe_cost, df_results, state, time_f, nsh = ds_battery_renewable (sol_try, 
                                                                                  demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir)
                
            elif (strategy_def in list_ds_dies_batt_renew):
                lcoe_cost, df_results, state, time_f, nsh = ds_dies_batt_renew(sol_try, 
                                                                               demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir)
                
            else:
                #no feasible combination
                state = 'no feasible'
                df_results = []
            
    
            print("finish simulation - state: " + state)
            #Create results
            if state == 'optimal':
                sol_try.results = Results(sol_try, df_results, lcoe_cost)
                sol_try.feasible = True
                sol_current = copy.deepcopy(sol_try)
                #Search the best solution
                if sol_try.results.descriptive['LCOE'] <= sol_best.results.descriptive['LCOE']:
                    #calculate area
                    sol_try.results.descriptive['area'] = calculate_area(sol_try)
                    #save sol_best
                    sol_best = copy.deepcopy(sol_try)   
                    best_nsh = nsh
                    
            else:
                sol_try.feasible = False
                df_results = []
                lcoe_cost = None
                sol_try.results = Results(sol_try, df_results, lcoe_cost)
                sol_current = copy.deepcopy(sol_try)
            
            #calculate area
            sol_current.results.descriptive['area'] = calculate_area(sol_current)
            #delete to avoid overwriting
            del df_results
            del sol_try
            
        if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
            print('Not Feasible solutions')
        else:
            #df with the feasible solutions
            df_iterations = pd.DataFrame(rows_df, columns=["i","feasible",
                                                           "area", "LCOE_actual","LCOE_Best","Movement"])
            #print results best solution
            print(sol_best.results.descriptive)
            print(sol_best.results.df_results)
            print('best solution number of not served hours: ' + str(best_nsh))
            generation_graph = sol_best.results.generation_graph(0, len(demand_df))
            plot(generation_graph)
            try:
                #stats
                percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(sol_best.batteries_dict_sol,
                                                                                       sol_best.generators_dict_sol, sol_best.results, demand_df)
            except KeyError:
                pass
            #calculate current COP   

            lcoe_cop = TRM * sol_best.results.descriptive['LCOE']
            #create Excel
            '''
            sol_best.results.df_results.to_excel("resultsolarbat.xlsx")         
            percent_df.to_excel("percentresultssolarbat.xlsx")
    
            '''
    else:
        print('No feasible solution, review data')
else:
    print('No feasible solution, solution need diesel generators or batteries')


