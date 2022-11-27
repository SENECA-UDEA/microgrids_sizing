# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

"""
from Multiyear.utilitiesmy import read_data, create_objects, create_technologies, calculate_area, calculate_energy, interest_rate
from Multiyear.utilitiesmy import fiscal_incentive, calculate_cost_data, calculate_multiyear_data, calculate_invertercost
from Multiyear.classesmy import Random_create
import pandas as pd 
from Multiyear.operatorsmy import Sol_constructor, Search_operator
from plotly.offline import plot
from Multiyear.dispatchmy import def_strategy, dies, B_plus_D_plus_Ren, D_plus_S_and_or_W, B_plus_S_and_or_W 
from Multiyear.dispatchmy import Results
import copy
pd.options.display.max_columns = None



#Set the seed for random
'''
seed = 42
'''
seed = None

rand_ob = Random_create(seed = seed)

#add and remove funtion
add_function = 'GRASP'
remove_function = 'RANDOM'


#data place
place = 'Providencia'


'''
place = 'San_Andres'
place = 'Puerto_Nar'
place = 'Leticia'
place = 'Test'
place = 'Oswaldo'
'''

#trm to current COP
TRM = 3910
#time not served best solution
best_nsh = 0

github_rute = 'https://raw.githubusercontent.com/SENECA-UDEA/microgrids_sizing/development/data/'
# file paths github
demand_filepath = github_rute + place+'/demand_'+place+'.csv' 
forecast_filepath = github_rute + place+'/forecast_'+place+'.csv' 
units_filepath = github_rute + place+'/parameters_'+place+'.json' 
instanceData_filepath = github_rute + place+'/instance_data_'+place+'.json' 
fiscalData_filepath = github_rute +'fiscal_incentive.json'
myearData_filepath = github_rute +'fiscal_incentive.json'
 
# file paths local
demand_filepath = "../data/"+place+"/demand_"+place+".csv"
forecast_filepath = "../data/"+place+"/forecast_"+place+".csv"
units_filepath = "../data/"+place+"/parameters_"+place+".json"
instanceData_filepath = "../data/"+place+"/instance_data_"+place+".json"

#fiscal Data
fiscalData_filepath = "../data/Cost/fiscal_incentive.json"

#cost Data
costData_filepath = "../data/Cost/parameters_cost.json"

#multiyear Data
myearData_filepath = "../data/Cost/multiyear.json"


# read data
demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data, my_data = read_data(demand_filepath,
                                                                                                        forecast_filepath,
                                                                                                        units_filepath,
                                                                                                        instanceData_filepath,
                                                                                                        fiscalData_filepath,
                                                                                                        costData_filepath,
                                                                                                        myearData_filepath)

#calculate multiyear data
demand_df, forecast_df = calculate_multiyear_data(demand_df_i, forecast_df_i, my_data, instance_data['years'])

#calulate parameters
amax =  instance_data['amax'] 
N_iterations = instance_data['N_iterations']
#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, instance_data, cost_data)
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

# Create objects and generation rule
generators_dict, batteries_dict,  = create_objects(generators,
                                                   batteries,  
                                                   forecast_df,
                                                   demand_df,
                                                   instance_data,
                                                   my_data)
#create technologies
technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)


#check diesel or batteries and at least one generator, for feasibility
if ('D' in technologies_dict.keys() or 'B' in technologies_dict.keys() and generators_dict != {}):
    #create the initial solution operator
    sol_constructor = Sol_constructor(generators_dict, 
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
                                                   my_data)
    
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
    search_operator = Search_operator(generators_dict, 
                                batteries_dict,
                                demand_df,
                                forecast_df)
    
    #check that first solution is feasible
    if (sol_best.results != None):
        for i in range(instance_data['N_iterations']):
            #create df to export results
            rows_df.append([i, sol_current.feasible, 
                            sol_current.results.descriptive['area'], 
                            sol_current.results.descriptive['LCOE'], 
                            sol_best.results.descriptive['LCOE'], movement])
            if sol_current.feasible == True:     
                # save copy as the last solution feasible seen
                sol_feasible = copy.deepcopy(sol_current) 
                # Remove a generator or battery from the current solution
                if (remove_function == 'GRASP'):
                    sol_try, remove_report = search_operator.removeobject(sol_current, delta)
                elif (remove_function == 'RANDOM'):
                    sol_try, remove_report = search_operator.removerandomobject(sol_current, rand_ob)
    
                movement = "Remove"
            else:
                #  Create list of generators that could be added
                list_available_bat, list_available_gen, list_tec_gen  = search_operator.available(sol_current, amax)
                if (list_available_gen != [] or list_available_bat != []):
                    # Add a generator or battery to the current solution
                    if (add_function == 'GRASP'):
                        sol_try, remove_report = search_operator.addobject(sol_current, list_available_bat, list_available_gen, list_tec_gen, remove_report, instance_data['fuel_cost'], rand_ob, delta)
                    elif (add_function == 'RANDOM'):
                        sol_try = search_operator.addrandomobject(sol_current, list_available_bat, list_available_gen, list_tec_gen,rand_ob)
                    movement = "Add"
                else:
                    # return to the last feasible solution
                    sol_current = copy.deepcopy(sol_feasible)
                    continue # Skip running the model and go to the begining of the for loop
    
            #review which dispatch strategy to use
            strategy_def = def_strategy(generators_dict = sol_try.generators_dict_sol,
                                        batteries_dict = sol_try.batteries_dict_sol) 
            
            #calculate inverter cost with installed generators
            #val = instance_data['inverter_cost']#first of the functions
            #instance_data['inverter cost'] = calculate_invertercost(sol_try.generators_dict_sol,sol_try.batteries_dict_sol,val)
            
    
            
            print("defined strategy")
            #run the dispatch strategy
            if (strategy_def == "diesel"):
                lcoe_cost, df_results, state, time_f, nsh  = dies(sol_try, demand_df, instance_data, cost_data, my_data)
            elif (strategy_def == "diesel - solar") or (strategy_def == "diesel - wind") or (strategy_def == "diesel - solar - wind"):
                lcoe_cost, df_results, state, time_f, nsh   = D_plus_S_and_or_W(sol_try, demand_df, instance_data, cost_data, delta, my_data)
            elif (strategy_def == "battery - solar") or (strategy_def == "battery - wind") or (strategy_def == "battery - solar - wind"):
                lcoe_cost, df_results, state, time_f, nsh   = B_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data, delta, rand_ob, my_data)
            elif (strategy_def == "battery - diesel - wind") or (strategy_def == "battery - diesel - solar") or (strategy_def == "battery - diesel - solar - wind"):
                lcoe_cost, df_results, state, time_f, nsh   = B_plus_D_plus_Ren(sol_try, demand_df, instance_data, cost_data, delta, rand_ob, my_data)
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
            df_iterations = pd.DataFrame(rows_df, columns=["i", "feasible", "area", "LCOE_actual", "LCOE_Best","Movement"])
            #print results best solution
            print(sol_best.results.descriptive)
            print(sol_best.results.df_results)
            print('best solution number of not served hours: ' + str(best_nsh))
            generation_graph = sol_best.results.generation_graph(0,len(demand_df))
            plot(generation_graph)
            try:
                #stats
                percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(sol_best.batteries_dict_sol, sol_best.generators_dict_sol, sol_best.results, demand_df)
            except KeyError:
                pass
            #calculate current COP   

            LCOE_COP = TRM * sol_best.results.descriptive['LCOE']
            #create Excel
            '''
            sol_best.results.df_results.to_excel("resultsolarbat.xlsx")         
            percent_df.to_excel("percentresultssolarbat.xlsx")
    
            '''
    else:
        print('No feasible solution, review data')
else:
    print('No feasible solution, solution need diesel generators or batteries')



# -*- coding: utf-8 -*-

