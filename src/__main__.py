# -*- coding: utf-8 -*-
import src.opt as opt
import src.GestionMR as GestionMR
import time
import pyomo.environ as pyo
'''
import sys
sys.path.append('../../')
from src.support.utilities import read_data, create_technologies
from src.support.utilities import calculate_energy, interest_rate
from src.support.utilities import fiscal_incentive, calculate_cost_data
from src.support.utilities import calculate_inverter_cost, create_excel
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


SEED = None

rand_ob = RandomCreate(seed = SEED)

#add and remove funtion
ADD_FUNCTION = 'GRASP'
REMOVE_FUNCTION = 'RANDOM'

#time not served best solution
best_nsh = 0


#trm to current COP
TRM = 3910

'''



import os
import click

@click.command()
@click.option('--weather_forecast', '-wf', default=None, type=str, help='Path of weather forecast data .csv file')
@click.option('--demand_forecast', '-df', default=None, type=str, help='Path of demand forecast data .csv file')
@click.option('--generation_units', '-gu', default=None, type=str, help='Path of generation units parameters .json file')

#poner datos por defecto, se puede json?
@click.option('--instance_data', '-id', default=None, type=str, help='Path of instance data .json file')
@click.option('--fiscal_incentive', '-gu', default=None, type=str, help='Path of fiscal incentive .json file')
@click.option('--multiyear', '-gu', default='.csv', type=str, help='Path of multiyear .json file')
@click.option('--parameters_cost', '-gu', default=None, type=str, help='Path of parameters cost .json file')
@click.option('--solver_name', '-sn', default='gurobi', help='Solver name to be use to solve the model; default = gurobi')
@click.option('--plot_results', '-plt', default=False, type=bool, help='Plot generation results')
@click.option('--base_file_name', '-bfn', default=None, help='Base name for .xlsx output file')
def main(weather_forecast, demand_forecast, generation_units, instance_data,
         fiscal_incentive, multiyear, parameters_cost,
         solver_name, plot_results, base_file_name):
    return main_func(weather_forecast, demand_forecast, generation_units, instance_data,
             fiscal_incentive, multiyear, parameters_cost,
             solver_name, plot_results, base_file_name)


def input_check(weather_forecast, demand_forecast, generation_units,
                instance_data, fiscal_incentive, multiyear, parameters_cost):
    if not weather_forecast:
        raise RuntimeError('You have to set a weather_forecast input file')
    elif not os.path.exists(weather_forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(weather_forecast))
    elif not demand_forecast:
        raise RuntimeError('You have to set a demand_forecast input file')
    elif not os.path.exists(demand_forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(demand_forecast))
    elif not generation_units:
        raise RuntimeError('You have to set a generation_units input file')
    elif not os.path.exists(generation_units):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(generation_units))
    elif not instance_data:
        raise RuntimeError('You have to set isntance_data input file')
    elif not os.path.exists(instance_data):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(instance_data))
    elif not fiscal_incentive:
        raise RuntimeError('You have to set a fiscal_incentive input file')
    elif not os.path.exists(fiscal_incentive):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(fiscal_incentive))
    elif not multiyear:
        raise RuntimeError('You have to set a multiyear input file')
    elif not os.path.exists(multiyear):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(multiyear))
    elif not parameters_cost:
        raise RuntimeError('You have to set a parameters_cost input file')
    elif not os.path.exists(parameters_cost):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(parameters_cost))


def main_func(weather_forecast, demand_forecast, generation_units, instance_data,
         fiscal_incentive, multiyear, parameters_cost,
         solver_name, plot_results, base_file_name):

    input_check(weather_forecast, demand_forecast, generation_units,
                    instance_data, fiscal_incentive, multiyear, parameters_cost)

    
    '''
    demand_df_i, forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data = utilities.read_data(demand_filepath,
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
    generators, batteries = utilities.calculate_cost_data(generators, batteries, 
                                                instance_data, cost_data)
    #Demand to be covered
    demand_df_i['demand'] = instance_data['demand_covered'] * demand_df_i['demand'] 

    #Calculate interest rate
    ir = utilities.interest_rate(instance_data['i_f'], instance_data['inf'])
    #Calculate CRF
    CRF = (ir * (1 + ir)**(instance_data['years'])) / ((1 + ir)**
                                                      (instance_data['years']) - 1)  


    #Calculate fiscal incentives
    delta = utilities.fiscal_incentive(fisc_data['credit'], 
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
    dem_week_vec, dem_weekend_vec = utilities.week_vector_data(demand_d,
                                                     instance_data["year_of_data"], forecast_df_i["day"][0])
    wind_vec = utililities.hour_data(forecast_w)
    sol_vecdni = utililities.hour_data(forecast_d)
    sol_vecdhi = utililities.hour_data(forecast_h)
    sol_vecghi = utililities.hour_data(forecast_g)
    #get the best distribution for each previous df   
    dem_week_dist = utililities.get_best_distribution(dem_week_vec) 
    dem_weekend_dist = utililities.get_best_distribution(dem_weekend_vec) 
    wind_dist = utililities.get_best_distribution(wind_vec) 
    sol_distdni = utililities.get_best_distribution(sol_vecdni) 
    sol_distdhi = utililities.get_best_distribution(sol_vecdhi) 
    sol_distghi = utililities.get_best_distribution(sol_vecghi) 
    #mean for triangular
    param = instance_data['fuel_cost']

    solutions = {}
    #save demand and forecast to best solution
    demand_scenarios = {}
    forecast_scenarios = {}
    fuel_scenarios = {}
    
    for scn in range(N_SCENARIOS):

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
            
            #calculate current COP   
            lcoe_cop = TRM * best_sol.results.descriptive['LCOE']
            
            #create excel
            create_excel(best_sol, percent_df, "stochastic", average_lcoe_best,
                         average_optimal_sol, 1)
            
            
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
                best0_nsh = nsh
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
                create_excel(best_sol0, percent_df0, "stochastic_origin",
                             average_lcoe_best, average_optimal_sol, 1)
            else:
                print("The best solution is not feasible in original data")
    
    '''
    forecast_df, demand = GestionMR.read_data(weather_forecast, demand_forecast, sepr=';')
    generators_dict, battery = GestionMR.create_generators(generation_units, forecast_df)

    model = opt.make_model(generators_dict, forecast_df, battery, demand)

    optimizer = pyo.SolverFactory(solver_name)

    timea = time.time()
    opt_results = optimizer.solve(model)
    execution_time = time.time() - timea

    if plot_results:
        raise RuntimeError("This feature is still under development.")

    term_cond = opt_results.solver.termination_condition
    if term_cond != pyo.TerminationCondition.optimal:
        print ("Termination condition={}".format(term_cond))
        raise RuntimeError("Optimization failed.")

    model_results = GestionMR.Results(model)
    
    if base_file_name:
        model_results.export_results(opt_results, base_file_name)
    else:
        return model_results



if __name__ == "__main__":
    main()
