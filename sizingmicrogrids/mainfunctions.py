# -*- coding: utf-8 -*-
"""
@author: scastellanos
"""
from sizingmicrogrids.utilities import create_technologies 
from sizingmicrogrids.utilities import calculate_area, calculate_energy
from sizingmicrogrids.utilities import fiscal_incentive
from sizingmicrogrids.utilities import create_objects, interest_rate, ils
from sizingmicrogrids.utilities import create_excel, calculate_sizing_cost
from sizingmicrogrids.utilities import create_multiyear_objects
from sizingmicrogrids.utilities import calculate_stochasticity_forecast
from sizingmicrogrids.utilities import generate_number_distribution
from sizingmicrogrids.utilities import calculate_stochasticity_demand
from sizingmicrogrids.utilities import  get_best_distribution, hour_data
from sizingmicrogrids.utilities import  week_vector_data, update_forecast
from sizingmicrogrids.utilities import calculate_multiyear_data
from sizingmicrogrids.strategies import dispatch_strategy, Results
from sizingmicrogrids.strategies import Results_my, dispatch_my_strategy
import math

import pandas as pd 
from sizingmicrogrids.operators import SolConstructor, SearchOperator
from plotly.offline import plot
import copy
import sizingmicrogrids.opt as opt
pd.options.display.max_columns = None



def maindispatch(demand_df, forecast_df, generators, batteries, instance_data,
                 fisc_data, cost_data,
                 best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path):
    '''
    This function makes a microgrid sizing optimization; 
    according to the two-stage formulation, 
    where iterated local search is done to install the elements, 
    and dispatch strategies to evaluate the performance.
    It starts by importing several data, as well as several functions 
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
    It also has other tools such as generating Excel files.

    Parameters
    ----------
    demand_df : DATAFRAME
        Data of year demand
    forecast_df : DATAFRAME
        Meteorological data
    generators : LIST
        List of all generators with their characteristics
    batteries : LIST
        List of all batteries with their characteristics
    instance_data : DICTIONARY
        Contains the data for the instance
    fisc_data : DICTIONARY
        Contains the data for the Tax incentive
    cost_data : DICTIONARY
        Contains information about costs of generarors or batteries
    best_nsh : Default = 0
        At the end the model reports the total of not served hours
    rand_ob : RANDOM CLASS
        Generates the random functions
    ADD_FUNCTION : STRING
        (GRASP or RANDOM) select the type of add function to applies
    REMOVE_FUNCTION : STRING
        (GRASP or RANDOM) select the type of remove function to applies
    folder_path : STRING
        Path where the user wants to save the file
    
    Returns
    -------
    percent_df : DATAFRAME
        Report on the generation percentages of the installed microgrid
    energy_df : DATAFRAME
        Report on the generation by each technology of the installed microgrid
    renew_df : DATAFRAME
        Report on the generation by renewable sources of the installed microgrid
    total_df : DATAFRAME
        Report on the total generation of the installed microgrid
    brand_df : DATAFRAME
        Report on the generation by each brand of the installed microgrid

    '''
    percent_df = {} 
    energy_df = {} 
    renew_df = {}
    total_df = {}
    brand_df = {}
    #calulate parameters
    AMAX = instance_data['amax'] 
    N_ITERATIONS = instance_data['N_iterations']

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
                                                                  rand_ob,
                                                                  cost_data,
                                                                  'DS',
                                                                  {},
                                                                  CRF,
                                                                  {},
                                                                  0)
        
        
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
                
                create_excel(sol_best, percent_df, "deterministic",folder_path,0 ,0 ,0)
                
                print("The output results are located in " + str(folder_path))

        else:
            print('No feasible solution, review data')
    else:
        print('No feasible solution, solution need diesel generators or batteries')

    return percent_df, energy_df, renew_df, total_df, brand_df 


def maindispatchmy(demand_df, forecast_df, generators, batteries, instance_data, 
                   fisc_data, cost_data,
                   my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path):
    
    '''
    This function performs a microgrid sizing optimization;
    according to the two-stage multiyear formulation, 
    where iterated local search is done to install the elements, 
    and dispatch strategies to evaluate the performance, 
    in addition, the multi-year functionality is added
    It starts by importing several data, as well as several functions 
    It specifies some parameters such as the seed for random and the location 
    Then it defines several lists of strategies for the optimization process.
    The script then goes on to perform microgrid optimization using the imported 
    functions and the data read in earlier. 
    It solves the model with the ILS and the dispatch strategy, with an aggregation 
    and disaggregation strategy and verifying feasibility.
    Finally, it uses the plotly library to create visualizations of the results of
    the best solution
    The code allows to change to different locations by uncommenting 
    the appropriate lines.
    It also has other tools such as generating Excel files 

    Parameters
    ----------
    demand_df : DATAFRAME
        Data of horizon time demand
    forecast_df : DATAFRAME
        Meteorological data
    generators : LIST
        List of all generators with their characteristics
    batteries : LIST
        List of all batteries with their characteristics
    instance_data : DICTIONARY
        Contains the data for the instance
    fisc_data : DICTIONARY
        Contains the data for the Tax incentive
    cost_data : DICTIONARY
        Contains information about costs of generarors or batteries
    my_data : DICTIONARY
        Contains information about the multiyear formulation
    best_nsh : Default = 0
        At the end the model reports the total of not served hours
    rand_ob : RANDOM CLASS
        Generates the random functions
    ADD_FUNCTION : STRING
        (GRASP or RANDOM) select the type of add function to applies
    REMOVE_FUNCTION : STRING
        (GRASP or RANDOM) select the type of remove function to applies
    folder_path : STRING
        Path where the user wants to save the file
    Returns
    -------
    percent_df : DATAFRAME
        Report on the generation percentages of the installed microgrid
    energy_df : DATAFRAME
        Report on the generation by each technology of the installed microgrid
    renew_df : DATAFRAME
        Report on the generation by renewable sources of the installed microgrid
    total_df : DATAFRAME
        Report on the total generation of the installed microgrid
    brand_df : DATAFRAME
        Report on the generation by each brand of the installed microgrid
        
    '''
    percent_df = {} 
    energy_df = {} 
    renew_df = {}
    total_df = {}
    brand_df = {}
    
    #calulate parameters
    AMAX = instance_data['amax'] 
    N_ITERATIONS = instance_data['N_iterations']
        
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
    generators_dict, batteries_dict = create_multiyear_objects(generators,
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
        sol_feasible, best_nsh = sol_constructor.initial_solution(instance_data,
                                                                  technologies_dict, 
                                                                  renewables_dict,
                                                                  delta,
                                                                  rand_ob,
                                                                  cost_data,
                                                                  'MY',
                                                                  {},0,
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
                                              demand_df, cost_data, 'ILS-DS-MY', best_nsh, 0, ir, my_data)
                
            if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
                print('Not Feasible solutions')
            else:
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
                #create Excel
                
                create_excel(sol_best, percent_df, "Multiyear", folder_path, 0, 0, 0)
                
                print("The output results are located in " + str(folder_path))
        else:
            print('No feasible solution, review data')
    else:
        print('No feasible solution, solution need diesel generators or batteries')
    
    return percent_df, energy_df, renew_df, total_df, brand_df



def mainstoc(demand_df_i, forecast_df_i, generators, batteries, instance_data,
             fisc_data, cost_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path):
    '''
    This function uses several procedures to perform
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
    robust and most economical.
     
    To solve, It uses the ILS and the dispatch strategy, with an aggregation 
    and disaggregation strategy and verifying feasibility.
    Finally, it uses the plotly library to create visualizations of the results of
    the best solution
    the code creates different scenarios from the probability distribution
    associated with each hourly interval of the data.
    To make the projection of demand, a distinction is made 
    between weekends and weekdays.
    The code allows to change to different locations by uncommenting 
    the appropriate lines.
    It also has other tools such as generating Excel files
 
    Parameters
    ----------
    demand_df_i : DATAFRAME
        Data of the initial year demand
    forecast_df_i : DATAFRAME
        Meteorological initial data
    generators : LIST
        list of all generators with their characteristics
    batteries : LIST
        list of all batteries with their characteristics
    instance_data : DICTIONARY
        Contains the data for the instance
    fisc_data : DICTIONARY
        Contains the data for the Tax incentive
    cost_data : DICTIONARY
        Contains information about costs of generarors or batteries
    best_nsh : Default = 0
        At the end the model reports the total of not served hours
    rand_ob : RANDOM CLASS
        Generates the random functions
    ADD_FUNCTION : STRING
        (GRASP or RANDOM) select the type of add function to applies
    REMOVE_FUNCTION : STRING
        (GRASP or RANDOM) select the type of remove function to applies
    folder_path : STRING
        Path where the user wants to save the file
        
    Returns
    -------
    percent_df : DATAFRAME
        Report on the generation percentages of the installed microgrid
    energy_df : DATAFRAME
        Report on the generation by each technology of the installed microgrid
    renew_df : DATAFRAME
        Report on the generation by renewable sources of the installed microgrid
    total_df : DATAFRAME
        Report on the total generation of the installed microgrid
    brand_df : DATAFRAME
        Report on the generation by each brand of the installed microgrid
    df_iterations : DATAFRAME
        Report the iterations of the ILS
    percent_df0 : DATAFRAME
        Report on the generation percentages of the installed microgrid 
        in the original data
    solutions : DICTIONARY
        Consolidation of the best solutions found
        in each scenario of the stochastic model

    ''' 
    percent_df = {} 
    percent_df0 = {} 
    energy_df = {} 
    renew_df = {}
    total_df = {}
    brand_df = {}
    df_iterations = {}
        
        
    #calulate parameters
    AMAX = instance_data['amax'] 
    N_ITERATIONS = instance_data['N_iterations']
    #number of scenarios
    N_SCENARIOS = instance_data['n-scenarios']
    
    
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
                                                                      rand_ob,
                                                                      cost_data,
                                                                      'DS',
                                                                      {},
                                                                      CRF,
                                                                      {},0)
            
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
                    print('Not Feasible solutions in scenario #' + str(scn))
                else:
                    #df with the feasible solutions
                    df_iterations = pd.DataFrame(rows_df, columns=["i","feasible","area", 
                                                                   "LCOE_actual","LCOE_Best","Movement"])
                    print('solution found in scenario #' + str(scn))
                    #save the solution
                    solutions[scn] = sol_best
            else:
                print('No feasible solution in scenario #' + str(scn))
        else:
            print('Solution need diesel generators or batteries, no feasible solution in scenario #' + str(scn))
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
                #update fuel cost
                instance_data['fuel_cost'] = fuel_scenarios[scn2]
                
                #update generation solar and wind
                generators_solution = solutions[scn].generators_dict_sol
                solutions[scn].generators_dict_sol = update_forecast(generators_solution, 
                                                                     forecast_scenarios[scn2], instance_data)
    
    
                #Run the dispatch strategy process
                lcoe_cost, df_results, state, time_f, nsh = dispatch_strategy(solutions[scn], 
                                                                              demand_scenarios[scn2], instance_data, cost_data, CRF, delta, rand_ob)
                #save the results
                if state == 'optimal':
                    sol_scn = copy.deepcopy(solutions[scn])
                    sol_scn.results = Results(sol_scn, df_results, lcoe_cost)
                    best_solutions[scn][scn2] = ['optimal',sol_scn.results.descriptive['LCOE'] ]  
                else:
                    best_solutions[scn][scn2] = ['No feasible', math.inf] 
        
        average = {}
        robust_solutions = {}
        optimal_len = {}
    
        
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
                optimal_len[i] = len(optimal_solutions)/N_SCENARIOS
            
            #select the best solution
            best_sol_position = min(average, key=average.get)       
            
            best_sol = solutions[best_sol_position]            
            average_lcoe_best = average[best_sol_position]
            average_optimal_sol = optimal_len[best_sol_position]
            
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
            
            #create excel
            create_excel(best_sol, percent_df, "stochastic", folder_path, average_lcoe_best,
                         average_optimal_sol, 1)
            
            print("The output results are located in " + str(folder_path))
            
            '''get the best solution in original data'''
            generators_0 = solutions[best_sol_position].generators_dict_sol
            #update generation solar and wind
            solutions[best_sol_position].generators_dict_sol = update_forecast(generators_0, 
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
                #calculate area
                best_sol0.results.descriptive['area'] = calculate_area(best_sol0)
               
                print(best_sol0.results.descriptive)
                print(best_sol0.results.df_results)
                #return number of no feasible solutions
                generation_graph = best_sol0.results.generation_graph(0, len(demand_scenarios[0]))
                plot(generation_graph)
                
                try:
                    #stats
                    percent_df0, energy_df, renew_df, total_df, brand_df = calculate_energy(best_sol0.batteries_dict_sol, 
                                                                                           best_sol0.generators_dict_sol, best_sol0.results, demand_scenarios[0])
                except KeyError:
                    pass
                create_excel(best_sol0, percent_df0, "stochastic_origin", folder_path,
                             average_lcoe_best, average_optimal_sol, 1)
                
                print("The output results in the original data are located in " + str(folder_path))
                
            else:
                print("The best solution is not feasible in original data")
    return percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions



def mainstocmy(demand_df_year, forecast_df_year, generators, 
               batteries, instance_data,fisc_data, cost_data, my_data, best_nsh, 
               rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path):
    '''
    This function uses several data and functions to perform
    microgrid sizing optimization; according to the two-stage formulation, 
    where iterated local search is done to install the elements, 
    and dispatch strategies to evaluate the performance, 
    in addition, the multi-year functionality is added
    It starts by importing several modules, as well as several functions 
    It specifies some parameters such as the seed for random and the location 
    Then it defines several lists of strategies for the optimization process.
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
    It also has other tools such as generating Excel files.

    demand_df_year : DATAFRAME
        Data of the initial year demand
    forecast_df_year : DATAFRAME
        Meteorological initial year data
    generators : LIST
        list of all generators with their characteristics
    batteries : LIST
        list of all batteries with their characteristics
    instance_data : DICTIONARY
        Contains the data for the instance
    fisc_data : DICTIONARY
        Contains the data for the Tax incentive
    cost_data : DICTIONARY
        Contains information about costs of generarors or batteries
    my_data : DICTIONARY
        Contains information about the multiyear formulation
    best_nsh : Default = 0
        At the end the model reports the total of not served hours
    rand_ob : RANDOM CLASS
        Generates the random functions
    ADD_FUNCTION : STRING
        (GRASP or RANDOM) select the type of add function to applies
    REMOVE_FUNCTION : STRING
        (GRASP or RANDOM) select the type of remove function to applies
    folder_path : STRING
        Path where the user wants to save the file

    Returns
    -------
    percent_df : DATAFRAME
        Report on the generation percentages of the installed microgrid
    energy_df : DATAFRAME
        Report on the generation by each technology of the installed microgrid
    renew_df : DATAFRAME
        Report on the generation by renewable sources of the installed microgrid
    total_df : DATAFRAME
        Report on the total generation of the installed microgrid
    brand_df : DATAFRAME
        Report on the generation by each brand of the installed microgrid
    df_iterations : DATAFRAME
        Report the iterations of the ILS
    percent_df0 : DATAFRAME
        Report on the generation percentages of the installed microgrid 
        in the original data
    solutions : DICTIONARY
        Consolidation of the best solutions found
        in each scenario of the stochastic model
    '''
    percent_df = {} 
    percent_df0 = {} 
    energy_df = {} 
    renew_df = {}
    total_df = {}
    brand_df = {}
    df_iterations = {}

        
    #calulate parameters
    AMAX = instance_data['amax'] 
    N_ITERATIONS = instance_data['N_iterations']
    #number of scenarios
    N_SCENARIOS = instance_data['n-scenarios']
    
    #Calculate interest rate
    ir = interest_rate(instance_data['i_f'], instance_data['inf'])
     
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
    demand_d = demand_df_year['demand']
    forecast_w = forecast_df_year['Wt']
    forecast_d = forecast_df_year['DNI']
    forecast_h = forecast_df_year['DHI']
    forecast_g = forecast_df_year['GHI']
    #Get the vector df od each hour
    dem_week_vec, dem_weekend_vec = week_vector_data(demand_d,
                                                     instance_data["year_of_data"], forecast_df_year["day"][0])
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
        demand_p = copy.deepcopy(demand_df_year)
        forecast_p = copy.deepcopy(forecast_df_year)
        #initial run is the original data
        if (scn == 0):
            demand_df_i = demand_df_year
            forecast_df_i = forecast_df_year
        else:
            #create stochastic df with distriburions
            demand_df_i = calculate_stochasticity_demand(rand_ob, demand_p, 
                                                       dem_week_dist, dem_weekend_dist,
                                                       instance_data["year_of_data"], forecast_df_year["day"][0])
            forecast_df_i = calculate_stochasticity_forecast(rand_ob, forecast_p, wind_dist, 
                                                           sol_distdni, sol_distdhi, sol_distghi)
        
        demand_df, forecast_df = calculate_multiyear_data(demand_df_i, forecast_df_i,
                                                          my_data, instance_data['years']) 
        demand_scenarios[scn] = demand_df
        forecast_scenarios[scn] = forecast_df
        # Create objects and generation rule
    
        if (scn >= 1):
            #calculate triangular to fuel cost
            #if scn = 1 use original data
            aux_fuel_cost = generate_number_distribution(rand_ob, param, limit)
            instance_data['fuel_cost'] = aux_fuel_cost
            fuel_scenarios[scn] = aux_fuel_cost
        else:
            fuel_scenarios[scn] = instance_data['fuel_cost']
            
        # Create objects and generation rule
        generators_dict, batteries_dict = create_multiyear_objects(generators,
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
            sol_feasible, best_nsh = sol_constructor.initial_solution(instance_data,
                                                                      technologies_dict, 
                                                                      renewables_dict,
                                                                      delta,
                                                                      rand_ob,
                                                                      cost_data,
                                                                      'MY',
                                                                      {},0,
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
                                                  demand_df, cost_data,'ILS-DS-MY', best_nsh, 0, ir, my_data)
    
                if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
                    print('Not Feasible solutions in scenario #' + str(scn))
                else:
                    #df with the feasible solutions
                    df_iterations = pd.DataFrame(rows_df, columns=["i","feasible",
                                                                   "area", "LCOE_actual","LCOE_Best","Movement"])
                    #save the solution
                    print('solution found in scenario #' + str(scn))
                    solutions[scn] = sol_best
            else:
                print('No feasible solution in scenario #' + str(scn))
        else:
            print('Solution need diesel generators or batteries, no feasible solution in scenario #' + str(scn))
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
        best_solutions = {k : [0] * int(len(solutions.keys())) 
                          for k in range(int(len(solutions.keys())))}
        
        #solve each solution in each scenario
        for scn in list_scn:
            #test current solution in all scenarios
            for scn2 in solutions.keys():
                #update fuel cost
                instance_data['fuel_cost'] = fuel_scenarios[scn2]
                #update generation solar and wind
                generators_solution = solutions[scn].generators_dict_sol
                solutions[scn].generators_dict_sol = update_forecast(generators_solution, 
                                                                     forecast_scenarios[scn2], instance_data)            
                #Run the dispatch strategy process
                lcoe_cost, df_results, state, time_f, nsh = dispatch_my_strategy(solutions[scn],
                                                                                 demand_scenarios[scn2], instance_data, cost_data, delta, rand_ob, my_data, ir)
    
                #save the results
                if state == 'optimal':
                    sol_scn = copy.deepcopy(solutions[scn])
                    sol_scn.results = Results_my(sol_scn, df_results, lcoe_cost)
                    best_solutions[scn][scn2] = ['optimal',sol_scn.results.descriptive['LCOE']]
                else:
                    best_solutions[scn][scn2] = ['No feasible', math.inf] 
        
        average = {}
        robust_solutions = {}
        optimal_len = {}
        
        #select the robust solutions
        #elects only those that have a proportion of "optimal" 
        #solutions greater than or equal to 0.6.
        robust_solutions = {i: best_solutions[i] for i in range(len(best_solutions)) 
                            if sum([1 for sub_sol in best_solutions[i] 
                                    if sub_sol[0] == "optimal"]) / len(best_solutions[i]) >= robust_parameter}
        
        #select the best solutions - average lowest cost
        if (robust_solutions == {}):
            print('No solution meets the robustness criterion, make the criterion more flexible')
        #select the best solutions - average lowest cost
        else:
            for i in robust_solutions.keys():
                optimal_solutions = [best_solutions[i][j][1] for j in robust_solutions.keys() 
                                     if best_solutions[i][j][0] == "optimal"]
                average[i] = sum(optimal_solutions) / len(optimal_solutions)
                optimal_len[i] = len(optimal_solutions)/N_SCENARIOS
            
            #select the best solution
            best_sol_position = min(average, key=average.get)       
            
            best_sol = solutions[best_sol_position]
            average_lcoe_best = average[best_sol_position]
            average_optimal_sol = optimal_len[best_sol_position]
            
            
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
            
            #create excel
            create_excel(best_sol, percent_df, "stochasticmy", average_lcoe_best,
                         folder_path, average_optimal_sol, 1)
            
            print("The output results are located in " + str(folder_path))
            
            '''get the best solution in original data'''
            
            generators_0 = solutions[best_sol_position].generators_dict_sol
            #update generation solar and wind
            solutions[best_sol_position].generators_dict_sol = update_forecast(generators_0, 
                                                                 forecast_scenarios[0], instance_data)
            #update fuel cost
            instance_data['fuel_cost'] = fuel_scenarios[0]
            
            #Run the dispatch strategy process
            lcoe_cost, df_results, state, time_f, nsh = dispatch_my_strategy(solutions[best_sol_position],
                                                                             demand_scenarios[0], instance_data, cost_data, delta, rand_ob, my_data, ir)
    
            #save the results
            if state == 'optimal':
                print('The best solution is feasible in the original data')
                best_sol0 = copy.deepcopy(solutions[best_sol_position])
                best_sol0.results = Results_my(best_sol0, df_results, lcoe_cost)
                #calculate area
                best_sol0.results.descriptive['area'] = calculate_area(best_sol0)
               
                print(best_sol0.results.descriptive)
                print(best_sol0.results.df_results)
                #return number of no feasible solutions
                generation_graph = best_sol0.results.generation_graph(0, len(demand_scenarios[0]))
                plot(generation_graph)
                
                try:
                    #stats
                    percent_df0, energy_df, renew_df, total_df, brand_df = calculate_energy(best_sol0.batteries_dict_sol, 
                                                                                           best_sol0.generators_dict_sol, best_sol0.results, demand_scenarios[0])
                except KeyError:
                    pass
                create_excel(best_sol0, percent_df0, "stochasticmy_origin",
                             folder_path, average_lcoe_best, average_optimal_sol, 1)
                
                print("The output results in the original data are located in " + str(folder_path))
            else:
                print("The best solution is not feasible in original data")
        
    return percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions


def mainopt(demand_df,forecast_df, generators, batteries, instance_data, 
            fisc_data, cost_data, solver_data, folder_path):
    '''
    This function uses several data and procedures 
    that are used to create and optimize a microgrid sizing system; for the
    one-stage approach, total optimization.
    The code reads in demand, forecast, and generators and batteries data,
    as well as instance and fiscal data, and uses them to calculate costs, 
    create objects, and define generation rules. 
     
    It then uses this data to create technologies and renewables sets, 
    and calculates the interest rate and fiscal incentives.
    Finally, it creates an optimization model using a solver (Gurobi), 
    and sets the MIP gap.
    The code also uses plotly to create offline plots with the optimization results 
    The code allows to change to different locations by uncommenting 
    the appropriate lines.
    It also has other tools such as generating Excel files.
    
    Parameters
    ----------
    demand_df : DATAFRAME
        Data of year demand
    forecast_df : DATAFRAME
        Meteorological data
    generators : LIST
        List of all generators with their characteristics
    batteries : LIST
        List of all batteries with their characteristics
    instance_data : DICTIONARY
        Contains the data for the instance
    fisc_data : DICTIONARY
        Contains the data for the Tax incentive
    cost_data : DICTIONARY
        Contains information about costs of generarors or batteries
    solver_data : DICTIONARY
        Contains information about the solver parameters.
    folder_path : STRING
        Path where the user wants to save the file
    
    Returns
    -------
    percent_df : DATAFRAME
        Report on the generation percentages of the installed microgrid
    energy_df : DATAFRAME
        Report on the generation by each technology of the installed microgrid
    renew_df : DATAFRAME
        Report on the generation by renewable sources of the installed microgrid
    total_df : DATAFRAME
        Report on the total generation of the installed microgrid
    brand_df : DATAFRAME
        Report on the generation by each brand of the installed microgrid

    '''
    percent_df = {} 
    energy_df = {} 
    renew_df = {}
    total_df = {}
    brand_df = {}
    
    # Create objects and generation rule
    generators_dict, batteries_dict = create_objects(generators,
                                                     batteries, 
                                                     forecast_df,
                                                     demand_df,
                                                     instance_data)
    
    #Create technologies and renewables set
    technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                              batteries_dict)
    
    
    #Calculate interest rate
    ir = interest_rate(instance_data['i_f'], instance_data['inf'])
    
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
                           fuel_cost = instance_data['fuel_cost'],
                           ir = ir, 
                           nse = instance_data['nse'], 
                           years = instance_data['years'],
                           splus_cost = instance_data['splus_cost'],
                           tlpsp = instance_data['tlpsp'],
                           delta = delta,
                           inverter = instance_data['inverter_cost'],
                           nse_cost = cost_data['NSE_COST'])    
    
    print("Model generated")
    # solve model 
    results, termination = opt.solve_model(model, 
                                            solver_data)
    print("Model optimized")
    
    if termination['Temination Condition'] == 'optimal': 
       #create results
       model_results = opt.Results(model, generators_dict, batteries_dict,'One-Stage')
       print(model_results.descriptive)
       print(model_results.df_results)
       generation_graph = model_results.generation_graph(0, len(demand_df))
       plot(generation_graph)
       try:
           #create stats
           percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(batteries_dict, 
                                                                                  generators_dict, model_results, demand_df)
       except KeyError:
           pass
    
    
    #Create Excel File
    percent_df.to_excel(str(folder_path) + "/" + "percentresults.xlsx")
    model_results.df_results.to_excel(str(folder_path) + "/" + "results.xlsx") 
    print("The output results are located in " + str(folder_path))
    return percent_df, energy_df, renew_df, total_df, brand_df


def mainopttstage (demand_df, forecast_df, generators, batteries,
                   instance_data, fisc_data,
                   cost_data, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, solver_data, folder_path):
    '''
    This functions import several data and procedures
    that are used to create and optimize a microgrid sizing system; for the 
    two-stage approach, where iterated local search is done to install the elements, 
    and operational optimization to evaluate the performance.
    The code reads in demand, forecast, and generators and batteries data,
    as well as instance and fiscal data, and uses them to calculate costs, 
    create objects, calculate the CRF to annualize the costs,
    among others. 
     
    It then uses this data to create technologies and renewables sets, 
    and calculates the interest rate and fiscal incentives.
    It solves the model with the ILS, with an aggregation and disaggregation 
    strategy, to evaluate the best performarnce solution, it creates an 
    optimization model using a solver (Gurobi), and sets the MIP gap.
    The code also uses plotly to create offline plots with the optimization results 
    The code allows to change to different locations by uncommenting 
    the appropriate lines.
    It also has other tools such as generating Excel files.
    
    Parameters
    ----------
    demand_df : DATAFRAME
        Data of year demand
    forecast_df : DATAFRAME
        Meteorological data
    generators : LIST
        List of all generators with their characteristics
    batteries : LIST
        List of all batteries with their characteristics
    instance_data : DICTIONARY
        Contains the data for the instance
    fisc_data : DICTIONARY
        Contains the data for the Tax incentive
    cost_data : DICTIONARY
        Contains information about costs of generarors or batteries
    rand_ob : RANDOM CLASS
        Generates the random functions
    ADD_FUNCTION : STRING
        (GRASP or RANDOM) select the type of add function to applies
    REMOVE_FUNCTION : STRING
        (GRASP or RANDOM) select the type of remove function to applies
    solver_data : DICTIONARY
        Contains information about the solver parameters.
    folder_path : STRING
        Path where the user wants to save the file
        
    Returns
    -------
    percent_df : DATAFRAME
        Report on the generation percentages of the installed microgrid
    energy_df : DATAFRAME
        Report on the generation by each technology of the installed microgrid
    renew_df : DATAFRAME
        Report on the generation by renewable sources of the installed microgrid
    total_df : DATAFRAME
        Report on the total generation of the installed microgrid
    brand_df : DATAFRAME
        Report on the generation by each brand of the installed microgrid
    df_iterations : DATAFRAME
        Report the iterations of the ILS

    '''
    percent_df = {} 
    energy_df = {} 
    renew_df = {}
    total_df = {}
    brand_df = {}
    df_iterations = {}
    #Inputs for the model
    AMAX = instance_data['amax']
    N_ITERATIONS = instance_data['N_iterations']

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

    #create a default solution, aux_variable is not relevant
    sol_feasible, aux_variable = sol_constructor.initial_solution(instance_data,
                                                                   technologies_dict, 
                                                                   renewables_dict,
                                                                   delta,
                                                                   rand_ob,
                                                                   cost_data,
                                                                   'OP',
                                                                   solver_data,
                                                                   0,{},0)

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
                                                                           delta, rand_ob, 'OP', CRF)
                    
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
                                                                            instance_data['fuel_cost'], rand_ob, delta,'OP',CRF)
                   

                    elif (ADD_FUNCTION == 'RANDOM'):
                        sol_try = search_operator.add_random_object(sol_current, 
                                                                    list_available_bat, list_available_gen, list_tec_gen, rand_ob)
                    
                    movement = "Add"
                else:
                    # return to the last feasible solution
                    sol_current = copy.deepcopy(sol_feasible)
                    continue # Skip running the model and go to the begining of the for loop

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
  
            #create Excel
            
            sol_best.results.df_results.to_excel(str(folder_path) + "/" +"results.xlsx")         
            percent_df.to_excel(str(folder_path) + "/" +"percentresults.xlsx")
            print("The output results are located in " + str(folder_path))
    else:
        print('No feasible solution, review data')
    return percent_df, energy_df, renew_df, total_df, brand_df, df_iterations