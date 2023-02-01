# -*- coding: utf-8 -*-
"""
This code imports several modules and sets up file paths for data files 
that are used to create and optimize a microgrid sizing system; for the 
two-stage approach, where iterated local search is done to install the elements, 
and operational optimization to evaluate the performance.

The code reads in demand, forecast, and generators and batteries data,
as well as instance and fiscal data, and uses them to calculate costs, 
create objects, set the random seed, calculate the CRF to annualize the costs,
among others. 
 
It then uses this data to create technologies and renewables sets, 
and calculates the interest rate and fiscal incentives.

It solves the model with the ILS, with an aggregation and disaggregation 
strategy, to evaluate the best performarnce solution, it creates an 
optimization model using a solver (Gurobi), and sets the MIP gap.

The code also uses plotly to create offline plots with the optimization results 

The code allows to change to different locations by uncommenting 
the appropriate lines.

It also has other tools such as generating Excel files or calculating the cost
 according to the representative market rate.

@author: scastellanos
"""
from src.support.utilities import read_data, calculate_sizing_cost
from src.support.utilities import create_objects, calculate_area
from src.support.utilities import create_technologies, calculate_energy
from src.support.utilities import interest_rate, calculate_inverter_cost
from src.support.utilities import fiscal_incentive, calculate_cost_data
import src.optimization.opt as opt
from src.support.classes import RandomCreate
import pandas as pd 
from src.optimization.operators import SolConstructor, SearchOperator
from plotly.offline import plot
import copy
pd.options.display.max_columns = None

#Algorythm data

#Set the seed for random

#SEED = 42

SEED = None
rand_ob = RandomCreate(seed = SEED)

#Set solver settings
solver_data = {"MIP_GAP":0.01,"TEE_SOLVER":True,"OPT_SOLVER":"gurobi"}
#select the search strategy
ADD_FUNCTION = 'GRASP'
REMOVE_FUNCTION = 'RANDOM'

#Model data
PLACE = 'Providencia'
#trm to current COP
TRM = 3910

'''
PLACE = 'San_Andres'
PLACE = 'Puerto_Nar'
PLACE = 'Leticia'
'''
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


# read data
demand_df, forecast_df, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                forecast_filepath,
                                                                                                units_filepath,
                                                                                                instanceData_filepath,
                                                                                                fiscalData_filepath,
                                                                                                costData_filepath)

#Inputs for the model
AMAX = instance_data['amax']
N_ITERATIONS = instance_data['N_iterations']

#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, 
                                            batteries, instance_data, cost_data)
#Demand to be covered
demand_df['demand'] = instance_data['demand_covered'] * demand_df['demand'] 

#Calculate interest rate
ir = interest_rate(instance_data['i_f'], instance_data['inf'])
#Calculate CRF
CRF = (ir * (1 + ir)**(instance_data['years'])) / ((1 + ir)
                                                   **(instance_data['years']) - 1)  

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
                                                 instance_data)
#create technologies
technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                          batteries_dict)

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
                                               solver_data,
                                               rand_ob,
                                               nse_cost = cost_data['NSE_COST'])

#if use aux_diesel asigns a big area to avoid select it again
if ('aux_diesel' in sol_feasible.generators_dict_sol.keys()):
    generators_dict['aux_diesel'] = sol_feasible.generators_dict_sol['aux_diesel']
    generators_dict['aux_diesel'].area = 10000000

# set the initial solution as the best so far
sol_best = copy.deepcopy(sol_feasible)

# create the actual solution with the initial soluion
sol_current = copy.deepcopy(sol_feasible)

#df of solutions
rows_df = []

# Create search operator
search_operator = SearchOperator(generators_dict, 
                                 batteries_dict,
                                 demand_df,
                                 forecast_df)

#check initial solution feasible
if (sol_best.results != None):
    movement = "Initial Solution"
    for i in range(N_ITERATIONS):
        '''
        ILS PROCEDURE
        '''
        #create data for df
        rows_df.append([i, sol_current.feasible, 
                        sol_current.results.descriptive['area'], 
                        sol_current.results.descriptive['LCOE'], 
                        sol_best.results.descriptive['LCOE'], movement])
        if sol_current.feasible:     
            # save copy as the last solution feasible seen
            sol_feasible = copy.deepcopy(sol_current) 
            # Remove a generator or battery from the current solution - grasp or random
            if (REMOVE_FUNCTION == 'GRASP'):
                sol_try, remove_report = search_operator.remove_object(sol_current, 
                                                                       CRF, delta, rand_ob)
                
            elif (REMOVE_FUNCTION == 'RANDOM'):
                sol_try, remove_report = search_operator.remove_random_object(sol_current, 
                                                                              rand_ob)

            movement = "Remove"
        else:
            #  Create list of generators that could be added
            list_available_bat, list_available_gen, list_tec_gen  = search_operator.available_items(sol_current,
                                                                                                    AMAX)
            
            if (list_available_gen != [] or list_available_bat != []):
                # Add a generator or battery to the current solution - grasp or random
                if (ADD_FUNCTION == 'GRASP'):
                    sol_try, remove_report = search_operator.add_object(sol_current, 
                                                                        list_available_bat, list_available_gen, list_tec_gen, remove_report,  
                                                                        CRF, instance_data['fuel_cost'], rand_ob, delta)
               
                elif (ADD_FUNCTION == 'RANDOM'):
                    sol_try = search_operator.add_random_object(sol_current, 
                                                                list_available_bat, list_available_gen, list_tec_gen, rand_ob)
                
                movement = "Add"
            else:
                # return to the last feasible solution
                sol_current = copy.deepcopy(sol_feasible)
                continue # Skip running the model and go to the begining of the for loop
        
        #calculate inverter cost with installed generators
        #val = instance_data['inverter_cost']#first of the functions
        #instance_data['inverter cost'] = calculate_inverter_cost(sol_try.generators_dict_sol,sol_try.batteries_dict_sol,val)
        
        #Calculate strategic cost
        tnpccrf_calc = calculate_sizing_cost(sol_try.generators_dict_sol, 
                                            sol_try.batteries_dict_sol, 
                                            ir = ir,
                                            years = instance_data['years'],
                                            delta = delta,
                                            inverter = instance_data['inverter_cost'])
        #Make model
        model = opt.make_model_operational(generators_dict = sol_try.generators_dict_sol,
                                           batteries_dict = sol_try.batteries_dict_sol,  
                                           demand_df = dict(zip(demand_df.t, demand_df.demand)), 
                                           technologies_dict = sol_try.technologies_dict_sol,  
                                           renewables_dict = sol_try.renewables_dict_sol,
                                           fuel_cost = instance_data['fuel_cost'],
                                           nse = instance_data['nse'], 
                                           TNPCCRF = tnpccrf_calc,
                                           splus_cost = instance_data['splus_cost'],
                                           tlpsp = instance_data['tlpsp'],
                                           nse_cost = cost_data['NSE_COST']) 
        
        #Solve the model
        results, termination = opt.solve_model(model, 
                                               solver_data)
        
        #Create results
        if termination['Temination Condition'] == 'optimal':
            sol_try.results.descriptive['LCOE'] = model.LCOE_value.expr()
            sol_try.results = opt.Results(model, sol_try.generators_dict_sol,
                                          sol_try.batteries_dict_sol,'Two-Stage')
            
            sol_try.feasible = True
            sol_current = copy.deepcopy(sol_try)
            #Search the best solution
            if sol_try.results.descriptive['LCOE'] <= sol_best.results.descriptive['LCOE']:
                sol_best = copy.deepcopy(sol_try)
                #calculate area best solution
                sol_best.results.descriptive['area'] = calculate_area(sol_best)
                
        else:
            sol_try.feasible = False
            sol_try.results.descriptive['LCOE'] = None
            sol_current = copy.deepcopy(sol_try)
        
        #avoid to overwrite an iteration
        del results            
        del termination
        del model 
        #calculate area solution
        sol_current.results.descriptive['area'] = calculate_area(sol_current)
    
    if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
        print('Not Feasible solutions')
    else:
        #df with the feasible solutions
        df_iterations = pd.DataFrame(rows_df, columns=["i","feasible", 
                                                       "area", "LCOE_actual","LCOE_Best","Movement"])
        #print results best solution
        print(sol_best.results.descriptive)
        print(sol_best.results.df_results)
        generation_graph = sol_best.results.generation_graph(0, len(demand_df))
        plot(generation_graph)
        try:
            #calculate stats
            percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(sol_best.batteries_dict_sol,
                                                                                   sol_best.generators_dict_sol, sol_best.results, demand_df)
        except KeyError:
            pass
        #calculate current COP   
       
        lcoe_cop = TRM * sol_best.results.descriptive['LCOE']
        #create Excel
        '''
        sol_best.results.df_results.to_excel("results2.xlsx")         
        percent_df.to_excel("percentresults.xlsx")

        '''
else:
    print('No feasible solution, review data')