# -*- coding: utf-8 -*-
"""
This Python script uses several imported modules and functions to perform
microgrid sizing optimization; according to the two-stage formulation, 
where iterated local search is done to install the elements, 
and dispatch strategies to evaluate the performance.

It starts by importing several modules, as well as several functions 

It specifies some parameters such as the seed for random and the location 
Then it defines several lists of strategies for the optimization process.
likewise, calculate the CRF to annualize the costs

The script then goes on to perform microgrid optimization using the imported 
functions and the data read in earlier. 

It solves the model with the ILS and the dispatch strategy, with an aggregation 
and disaggregation strategy and verifying feasibility.

Finally, it uses the plotly library to create visualizations of the results of
the best solution

The code allows to change to different locations by uncommenting 
the appropriate lines.

It also has other tools such as generating Excel files or calculating the cost
 according to the representative market rate.

"""
from src.support.utilities import read_data, create_technologies 
from src.support.utilities import calculate_area, calculate_energy
from src.support.utilities import fiscal_incentive, calculate_cost_data
from src.support.utilities import create_objects, interest_rate, ils
from src.support.utilities import create_excel
from src.support.classes import RandomCreate
import pandas as pd 
from src.simulation.operatorsdispatch import SolConstructor, SearchOperator
from plotly.offline import plot
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
#Calculate CRF
CRF = (ir * (1 + ir)**(instance_data['years'])) / ((1 + ir)**
                                                  (instance_data['years']) - 1)  

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


#check diesel or batteries and at least one generator, for feasibility
if ('D' in technologies_dict.keys() or 'B' in technologies_dict.keys() 
    and generators_dict != {}):
    #create the initial solution operator
    sol_constructor = SolConstructor(generators_dict, 
                                     batteries_dict,
                                     demand_df,
                                     forecast_df)
    
    #create a default solution
    sol_feasible, best_nsh = sol_constructor.initial_solution(instance_data,
                                                              technologies_dict, 
                                                              renewables_dict,
                                                              delta,
                                                              CRF,
                                                              rand_ob,
                                                              cost_data)
    
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
        sol_best, best_nsh, rows_df = ils(N_ITERATIONS, sol_current, sol_best,
                                          search_operator, REMOVE_FUNCTION, 
                                          ADD_FUNCTION,delta, rand_ob, instance_data, AMAX,
                                          demand_df, cost_data,'ILS-DS', best_nsh, CRF, 0, {})
            
        if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
            print('Not Feasible solutions')
        else:
            #df with the feasible solutions
            df_iterations = pd.DataFrame(rows_df, columns=["i","feasible","area", 
                                                           "LCOE_actual","LCOE_Best","Movement"])
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
            
            
            create_excel(sol_best, percent_df, "deterministic",0 ,0 ,0)
            

    else:
        print('No feasible solution, review data')
else:
    print('No feasible solution, solution need diesel generators or batteries')



