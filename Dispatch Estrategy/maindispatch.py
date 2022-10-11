# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""
from utilities import read_data, create_objects, calculate_sizingcost, create_technologies, calculate_area, calculate_energy, interest_rate
from utilities import fiscal_incentive, calculate_cost_data
#import opt as opt
from classes import Random_create
import pandas as pd 
import operatorsdispatch, graphic_results
import maindispatch
from operators import Sol_constructor, Search_operator
from plotly.offline import plot
from dispatchstrategy import strategy, d, bd, B_plus_D_plus_Ren, D_plus_S_and_or_W, B_plus_S_and_or_W 
import copy
pd.options.display.max_columns = None


place = 'Providencia'
place = 'Test'

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

#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, instance_data, cost_data)
#Demand to be covered
demand_df['demand'] = instance_data['demand_covered']  * demand_df['demand'] 

#Calculate interest rate
ir = interest_rate(instance_data['i_f'],instance_data['inf'])
#Calculate CRF
CRF = (ir * (1 + ir)**(instance_data['years']))/((1 + ir)**(instance_data['years'])-1)  


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
                                                   instance_data)
#create technologies
technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)

#create the initial solution operator
sol_constructor = Sol_constructor(generators_dict, 
                            batteries_dict,
                            demand_df,
                            forecast_df)

#create a default solution
sol_feasible = sol_constructor.initial_solution(instance_data,
                                               generators_dict, 
                                               batteries_dict, 
                                               technologies_dict, 
                                               renewables_dict,
                                               delta,
                                               rand_ob,
                                               lcoe_cost = cost_data['LCOE_COST'])

#if use aux_diesel asigns a big area to avoid select it again
if ('aux_diesel' in sol_feasible.generators_dict_sol.keys()):
    generators_dict['aux_diesel'] = sol_feasible.generators_dict_sol['aux_diesel']
    generators_dict['aux_diesel'].area = 10000000


# set the initial solution as the best so far
sol_best = copy.deepcopy(sol_feasible)

# create the actual solution with the initial soluion
sol_current = copy.deepcopy(sol_feasible)

#nputs for the model
movement = "Initial Solution"
amax =  instance_data['amax']
N_iterations = instance_data['N_iterations']
#df of solutions
rows_df = []

# Create search operator
search_operator = Search_operator(generators_dict, 
                            batteries_dict,
                            demand_df,
                            forecast_df)
add_function = 'GRASP'
remove_function = 'RANDOM'
if (sol_best.results != None):
    for i in range(N_iterations):
        rows_df.append([i, sol_current.feasible, 
                        sol_current.results.descriptive['area'], 
                        sol_current.results.descriptive['LCOE'], 
                        sol_best.results.descriptive['LCOE'], movement])
        if sol_current.feasible == True:     
            # save copy as the last solution feasible seen
            sol_feasible = copy.deepcopy(sol_current) 
            # Remove a generator or battery from the current solution
            if (remove_function == 'GRASP'):
                sol_try, remove_report = search_operator.removeobject(sol_current, CRF, delta)
            elif (remove_function == 'RANDOM'):
                sol_try, remove_report = search_operator.removerandomobject(sol_current, rand_ob)

            movement = "Remove"
        else:
            #  Create list of generators that could be added
            list_available_bat, list_available_gen, list_tec_gen  = search_operator.available(sol_current, amax)
            if (list_available_gen != [] or list_available_bat != []):
                # Add a generator or battery to the current solution
                if (add_function == 'GRASP'):
                    sol_try, remove_report = search_operator.addobject(sol_current, list_available_bat, list_available_gen, list_tec_gen, remove_report,  CRF, instance_data['fuel_cost'], rand_ob, delta)
                elif (add_function == 'RANDOM'):
                    sol_try = search_operator.addrandomobject(sol_current, list_available_bat, list_available_gen, list_tec_gen,rand_ob)
                movement = "Add"
            else:
                # return to the last feasible solution
                sol_current = copy.deepcopy(sol_feasible)
                continue # Skip running the model and go to the begining of the for loop

        #Make model
        
        strategy = strategy(generators_dict = sol_try.generators_dict_sol,
                                           batteries_dict = sol_try.batteries_dict_sol) 
        
        ar = calculate_area[sol_try]
        if ar > amax:
            state = 'No feasible solution'
        else:
            if (strategy == "Diesel"):
                lcoe_cost, df_results, state = d(sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "diesel - solar"):
                lcoe_cost, df_results, state = D_plus_S_and_or_W(sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery - diesel"):
                lcoe_cost, df_results, state = bd(sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery - solar"):
                lcoe_cost, df_results, state = B_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery - wind"):
                lcoe_cost, df_results, state = B_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "diesel - wind"):
                lcoe_cost, df_results, state = D_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery - solar - wind"):
                lcoe_cost, df_results, state = B_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery - diesel - wind"):
                lcoe_cost, df_results, state = B_plus_D_plus_Ren(sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "diesel - solar - wind"):
                lcoe_cost, df_results, state = D_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery diesel - solar"):
                lcoe_cost, df_results, state = B_plus_D_plus_Ren(sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            elif (strategy == "battery - diesel - solar - wind"):
                lcoe_cost, df_results, state = B_plus_D_plus_Ren(sol_try, demand_df, instance_data, cost_data['LCOE_COST'] )
            else:
                state = 'No feasible solution'
        

        #Create results
        if state == 'optimal':
            sol_try.results.descriptive['LCOE'] = lcoe_cost
            sol_try.results = df_results
            sol_try.feasible = True
            sol_current = copy.deepcopy(sol_try)
            #Search the best solution
            if sol_try.results.descriptive['LCOE'] <= sol_best.results.descriptive['LCOE']:
                sol_best = copy.deepcopy(sol_try)
        else:
            sol_try.feasible = False
            sol_try.results.descriptive['LCOE'] = None
            sol_current = copy.deepcopy(sol_try)
        
        del df_results
    
        sol_current.results.descriptive['area'] = calculate_area(sol_current)
    
    if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
        print('Not Feasible solutions')
    else:
        #df with the feasible solutions
        df_iterations = pd.DataFrame(rows_df, columns=["i", "feasible", "area", "LCOE_actual", "LCOE_Best","Movement"])
        #print results best solution
        print(sol_best.results.descriptive)
        print(sol_best.results.df_results)
        generation_graph = sol_best.results.generation_graph(0,len(demand_df))
        plot(generation_graph)
        try:
            percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(sol_best.batteries_dict_sol, sol_best.generators_dict_sol, sol_best.results, demand_df)
        except KeyError:
            pass
        #calculate current COP   
        TRM = 3910
        LCOE_COP = TRM * sol_best.results.descriptive['LCOE']
        #create Excel
        '''
        sol_best.results.df_results.to_excel("results2.xlsx")         
        percent_df.to_excel("percentresults.xlsx")

        '''
else:
    print('No feasible solution, review data')

