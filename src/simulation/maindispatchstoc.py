# -*- coding: utf-8 -*-
"""
This Python script uses several imported modules and functions to perform
microgrid sizing optimization; according to the two-stage formulation, 
where iterated local search is done to install the elements, 
and dispatch strategies to evaluate the performance and add stochasticity 
to go from a deterministic model to one with uncertainty

It starts by importing several modules, as well as several functions 

It specifies some parameters such as the seed for random and the location 
Then it defines several lists of strategies for the optimization process.
likewise, calculate the CRF to annualize the costs

The script then goes on to perform microgrid optimization using the imported 
functions and the data read in earlier. 

Solve the model for each scenario and then determine the best solution, 
testing all the solutions in the scenarios to determine which is the most 
robust and most economical
 
to solve, It uses the ILS and the dispatch strategy, with an aggregation 
and disaggregation strategy and verifying feasibility.

Finally, it uses the plotly library to create visualizations of the results of
the best solution

the code creates different scenarios from the probability distribution
associated with each hourly interval of the data.

To make the projection of demand, a distinction is made 
between weekends and weekdays.

The code allows to change to different locations by uncommenting 
the appropriate lines.

It also has other tools such as generating Excel files or calculating the cost
 according to the representative market rate.


"""
from src.support.utilities import read_data, create_technologies
from src.support.utilities import calculate_energy, interest_rate
from src.support.utilities import fiscal_incentive, calculate_cost_data
from src.support.utilities import calculate_inverter_cost
from src.support.utilities import calculate_stochasticity_forecast
from src.support.utilities import generate_number_distribution
from src.support.utilities import calculate_stochasticity_demand
from src.support.utilities import create_objects, calculate_area
from src.support.utilities import  get_best_distribution, hour_data
from src.support.utilities import  week_vector_data, update_forecast, ils

from src.support.classes import RandomCreate
import pandas as pd 
from src.simulation.operatorsdispatch import SolConstructor, SearchOperator
from plotly.offline import plot
from src.simulation.strategies import dispatch_strategy, Results
import copy
import math
pd.options.display.max_columns = None

#Set the seed for random

#SEED = 42

SEED = None

rand_ob = RandomCreate(seed = SEED)

#add and remove funtion
ADD_FUNCTION = 'GRASP'
REMOVE_FUNCTION = 'RANDOM'

#time not served best solution
best_nsh = 0

#data PLACE
PLACE = 'Providencia'

'''
PLACE = 'San_Andres'
PLACE = 'Puerto_Nar'
PLACE = 'Leticia'
'''

#trm to current COP
TRM = 3910

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
demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                forecast_filepath,
                                                                                                units_filepath,
                                                                                                instanceData_filepath,
                                                                                                fiscalData_filepath,
                                                                                                costData_filepath)

#calulate parameters
AMAX = instance_data['amax'] 
N_ITERATIONS = instance_data['N_iterations']
#number of scenarios
N_SCENARIOS = instance_data['n-scenarios']
#Calculate salvage, operation and replacement cost with investment cost
generators, batteries = calculate_cost_data(generators, batteries, 
                                            instance_data, cost_data)
#Demand to be covered
demand_df_i['demand'] = instance_data['demand_covered'] * demand_df_i['demand'] 

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

#STOCHASTICITY

#range for triangular distribution fuel cost
limit = instance_data['stochastic_fuel_cost']

#percent robustness
robust_parameter = instance_data['percent_robustness']

#percent robustness
sample_scenarios = instance_data['sample_escenarios']

#get the df for each hour and each data
demand_d = demand_df_i['demand']
forecast_w = forecast_df_i['Wt']
forecast_d = forecast_df_i['DNI']
forecast_h = forecast_df_i['DHI']
forecast_g = forecast_df_i['GHI']
#Get the vector df od each hour
dem_week_vec, dem_weekend_vec = week_vector_data(demand_d,
                                                 instance_data["year_of_data"], forecast_df_i["day"][0])
wind_vec = hour_data(forecast_w)
sol_vecdni = hour_data(forecast_d)
sol_vecdhi = hour_data(forecast_h)
sol_vecghi = hour_data(forecast_g)
#get the best distribution for each previous df   
dem_week_dist = get_best_distribution(dem_week_vec) 
dem_weekend_dist = get_best_distribution(dem_weekend_vec) 
wind_dist = get_best_distribution(wind_vec) 
sol_distdni = get_best_distribution(sol_vecdni) 
sol_distdhi = get_best_distribution(sol_vecdhi) 
sol_distghi = get_best_distribution(sol_vecghi) 
#mean for triangular
param = instance_data['fuel_cost']

solutions = {}
#save demand and forecast to best solution
demand_scenarios = {}
forecast_scenarios = {}
fuel_scenarios = {}
#scenarios       
for scn in range(N_SCENARIOS):
    '''
    Simulation by scenarios
    '''
    demand_p = copy.deepcopy(demand_df_i)
    forecast_p = copy.deepcopy(forecast_df_i)
    #initial run is the original data
    if (scn == 0):
        demand_df = demand_df_i
        forecast_df = forecast_df_i
    else:
        #create stochastic df with distriburions
        demand_df = calculate_stochasticity_demand(rand_ob, demand_p, 
                                                   dem_week_dist, dem_weekend_dist,
                                                   instance_data["year_of_data"], forecast_df_i["day"][0])
        forecast_df = calculate_stochasticity_forecast(rand_ob, forecast_p, wind_dist, 
                                                       sol_distdni, sol_distdhi, sol_distghi)
    
    demand_scenarios[scn] = demand_df
    forecast_scenarios[scn] = forecast_df
    # Create objects and generation rule
    generators_dict, batteries_dict = create_objects(generators,
                                                     batteries,  
                                                     forecast_df,
                                                     demand_df,
                                                     instance_data)

    #create technologies
    technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                              batteries_dict)
    if (scn >= 1):
        #calculate triangular to fuel cost
        #if scn = 1 use original data
        aux_fuel_cost = generate_number_distribution(rand_ob, param, limit)
        instance_data['fuel_cost'] = aux_fuel_cost
        fuel_scenarios[scn] = aux_fuel_cost
    else:
        fuel_scenarios[scn] = instance_data['fuel_cost']
        
    #check diesel or batteries and at least one generator, for feasibility
    if ('D' in technologies_dict.keys() or 'B' 
        in technologies_dict.keys() and generators_dict != {}):
        
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
                #save the solution
                solutions[scn] = sol_best
        else:
            print('No feasible solution, review data')
    else:
        print('No feasible solution, solution need diesel generators or batteries')
    del demand_df
    del forecast_df


if solutions == {}:
    print('there is no feasible solution in any scenario')

else:
    #get a sample if number of scenarios is higger than sample allowed   
    if (len(solutions.keys()) >= sample_scenarios):
        #select the best solutions - lowest LCOE
        list_scn = sorted(solutions.keys(), key = lambda scn: solutions[scn].results.descriptive['LCOE'])
        list_scn = list_scn[:sample_scenarios]
    else:
        list_scn = list(solutions.keys())
    
    #create matrix to save solutions
    best_solutions = {k : [0] * int(len(solutions.keys())) for k in range(int(len(solutions.keys())))}
    
    #solve each solution in each scenario
    for scn in list_scn:
        #test current solution in all scenarios
        for scn2 in solutions.keys():
            generators = solutions[scn].generators_dict_sol
            #update generation solar and wind
            solutions[scn].generators_dict_sol = update_forecast(generators, 
                                                                 forecast_scenarios[scn2], instance_data)
            #update fuel cost
            instance_data['fuel_cost'] = fuel_scenarios[scn2]

            #Run the dispatch strategy process
            lcoe_cost, df_results, state, time_f, nsh = dispatch_strategy(solutions[scn], 
                                                                          demand_scenarios[scn2], instance_data, cost_data, CRF, delta, rand_ob)
            #save the results
            if state == 'optimal':
                sol_current = copy.deepcopy(solutions[scn])
                sol_current.results = Results(sol_current, df_results, lcoe_cost)
                best_solutions[scn][scn2] = ['optimal',sol_current.results.descriptive['LCOE'] ]  
            else:
                best_solutions[scn][scn2] = ['No feasible', math.inf] 
    
    average = {}
    robust_solutions = {}
    
    #select the robust solutions
    #elects only those that have a proportion of "optimal" 
    #solutions greater than or equal to the criteria
    robust_solutions = {i: best_solutions[i] for i in range(len(best_solutions)) 
                        if sum([1 for sub_sol in best_solutions[i] 
                                if sub_sol[0] == "optimal"]) / len(best_solutions[i]) >= robust_parameter}
    
    if (robust_solutions == {}):
        print('No solution meets the robustness criterion, make the criterion more flexible')
    #select the best solutions - average lowest cost
    else:
        for i in robust_solutions.keys():
            optimal_solutions = [best_solutions[i][j][1] for j in robust_solutions.keys() 
                                 if best_solutions[i][j][0] == "optimal"]
            average[i] = sum(optimal_solutions) / len(optimal_solutions)
        
        #select the best solution
        best_sol_position = min(average, key=average.get)       
        
        best_sol = solutions[best_sol_position]
        
        #return the selected solution
        print(best_sol.results.descriptive)
        print(best_sol.results.df_results)
        #return number of no feasible solutions
        generation_graph = best_sol.results.generation_graph(0, len(demand_df_i))
        plot(generation_graph)
        
        try:
            #stats
            percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(best_sol.batteries_dict_sol, 
                                                                                   best_sol.generators_dict_sol, best_sol.results, demand_scenarios[best_sol_position])
        except KeyError:
            pass
        
        #calculate current COP   
        lcoe_cop = TRM * best_sol.results.descriptive['LCOE']
        
        '''get the best solution in original data'''
        generators = solutions[best_sol_position].generators_dict_sol
        #update generation solar and wind
        solutions[best_sol_position].generators_dict_sol = update_forecast(generators, 
                                                             forecast_scenarios[0], instance_data)
       
        #update fuel cost
        instance_data['fuel_cost'] = fuel_scenarios[0]
        
        #Run the dispatch strategy process
        lcoe_cost, df_results, state, time_f, nsh = dispatch_strategy(solutions[best_sol_position], 
                                                                      demand_scenarios[0], instance_data, cost_data, CRF, delta, rand_ob)
        #save the results
        if state == 'optimal':
            print('The best solution is feasible in the original data')
            best_sol0 = copy.deepcopy(solutions[best_sol_position])
            best_sol0.results = Results(best_sol0, df_results, lcoe_cost)
            best0_nsh = nsh
           
            print(best_sol0.results.descriptive)
            print(best_sol0.results.df_results)
            #return number of no feasible solutions
            generation_graph = best_sol0.results.generation_graph(0, len(demand_scenarios[0]))
            plot(generation_graph)
            
            try:
                #stats
                percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(best_sol0.batteries_dict_sol, 
                                                                                       best_sol0.generators_dict_sol, best_sol0.results, demand_scenarios[0])
            except KeyError:
                pass
        else:
            print("The best solution is not feasible in original data")
        '''
        #create Excel
        sol_best.results.df_results.to_excel("resultsolarbat.xlsx")         
        percent_df.to_excel("percentresultssolarbat.xlsx")
        
        '''