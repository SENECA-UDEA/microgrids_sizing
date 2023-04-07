
# -*- coding: utf-8 -*-
import mainfunctions as mf
from utilities import read_data, calculate_cost_data
from utilities import read_multiyear_data, calculate_multiyear_data
from classes import RandomCreate
import copy
import os
import click




'''Default location'''
PLACE = 'Providencia'
# file paths 
demand_filepath = "../data/" + PLACE + "/demand_" + PLACE+".csv"
forecast_filepath = "../data/"+PLACE+"/forecast_" + PLACE + ".csv"
units_filepath = "../data/" + PLACE + "/parameters_" + PLACE + ".json"
instanceData_filepath = "../data/" + PLACE + "/instance_data_" + PLACE + ".json"
#fiscal Data
fiscalData_filepath = "../data/auxiliar/fiscal_incentive.json"
#cost Data
costData_filepath = "../data/auxiliar/parameters_cost.json"
#multiyear Data
myearData_filepath = "../data/auxiliar/multiyear.json"


#default seed
SEED = None

#default folder path
default_folder_path = os.getcwd()

type_model_options = ['st','dt','my','sm','op','ot']


@click.command()
@click.option('--type_model', '-tm', default='st', type=str, help=
              'type of model {st = stochastic, dt = deterministic, my = multiyear, sm = stochastic multiyear, op = optimization one stage, ot = optimization two stage}')
@click.option('--demand', '-df', default=demand_filepath,
              type=str, help='Path of demand forecast data .csv file')
@click.option('--forecast', '-sw', default=forecast_filepath,
              type=str, help='Path of weather forecast data .csv file')
@click.option('--generation_units', '-gu', default=units_filepath,
              type=str, help='Path of generation and batteries units parameters .json file')
@click.option('--instance_filepath', '-id', default=instanceData_filepath,
              type=str, help='Path of instance data .json file')
@click.option('--tax_incentive', '-ft', default=fiscalData_filepath,
              type=str, help='Path of fiscal-tax incentive .json file')
@click.option('--parameters_cost', '-cd', default=costData_filepath,
              type=str, help='Path of parameters cost .json file')
@click.option('--rand_seed', '-rs', default=SEED, 
              type=int, help='Seed value for the random object')
@click.option('--folder_path', '-fp', default=default_folder_path,
              type=str,help='Base name for .xlsx output file')
@click.option('--my_data', '-md', default=myearData_filepath,
              type=str, help='Path of multiyear parameters .json file')
@click.option('--solver_name', '-sn', default='gurobi', 
              help='Solver name to be used to solve the model; default = gurobi')
@click.option('--gap', '-gp', default=0.01, type = float, 
              help='Solver GAP to be used to solve the model; default = 0.01')


def main (type_model, demand, forecast, generation_units, instance_filepath, tax_incentive, 
          parameters_cost, rand_seed, folder_path, my_data, solver_name, gap):
    return main_func(type_model, demand, forecast, generation_units, instance_filepath, 
                     tax_incentive, parameters_cost, rand_seed, folder_path, my_data, solver_name, gap)



def input_check(type_model, forecast, demand, generation_units):
    if not type_model:
        raise RuntimeError('You have to set a model type')
    elif not type_model in type_model_options:
        raise RuntimeError('You have to select a correct model type')
    elif not forecast:
        raise RuntimeError('You have to set a forecast input file')
    elif not os.path.exists(forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(forecast))
    elif not demand:
        raise RuntimeError('You have to set a demand input file')
    elif not os.path.exists(demand):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(demand))
    elif not generation_units:
        raise RuntimeError('You have to set a generation_units input file')
    elif not os.path.exists(generation_units):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(generation_units))



def main_func (type_model, demand, forecast, generation_units, instance_filepath, tax_incentive, 
              parameters_cost, rand_seed, folder_path, my_data, solver_name, gap):

    input_check(type_model, demand, forecast, generation_units)
    
    #Add and remove parameters
    ADD_FUNCTION = 'GRASP'
    REMOVE_FUNCTION = 'RANDOM'
    SEED = rand_seed
    rand_ob = RandomCreate(seed = SEED)
    
    #auxiliar best nsh
    best_nsh = 0
    
    #solve the model according to the selected type
    
    if (type_model != 'my' and type_model != 'sm'):
    
        # read data
        demand_df, forecast_df, generators, batteries, instance_data, fisc_data, cost_data = read_data(demand,
                                                                                                        forecast,
                                                                                                        generation_units,
                                                                                                        instance_filepath,
                                                                                                        tax_incentive,
                                                                                                        parameters_cost)

        #Calculate salvage, operation and replacement cost with investment cost
        generators, batteries = calculate_cost_data(generators, batteries, 
                                                    instance_data, cost_data)
        #Demand to be covered
        demand_df['demand'] = instance_data['demand_covered'] * demand_df['demand'] 
        
        if (type_model == 'dt'):
            percent_df, energy_df, renew_df, total_df, brand_df  = mf.maindispatch(demand_df,
                                                                                   forecast_df, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                   best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)
      
        elif (type_model == 'st'):
            demand_df_i = copy.deepcopy(demand_df)
            forecast_df_i = copy.deepcopy(forecast_df)
            percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions = mf.mainstoc(demand_df_i, 
                                                                                                                     forecast_df_i, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                                                     best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)
    
    
        elif (type_model == 'op' or type_model == 'ot'):
            #Set GAP
            #User selects gap and solver
            solver_data = {"MIP_GAP":gap,"TEE_SOLVER":True,"OPT_SOLVER":solver_name}
            
            if (type_model == 'op'):
                percent_df, energy_df, renew_df, total_df, brand_df  = mf.mainopt(demand_df,
                                                                                  forecast_df, generators, batteries, 
                                                                                  instance_data, fisc_data, cost_data, solver_data, folder_path)
            else:
                percent_df, energy_df, renew_df, total_df, brand_df, df_iterations = mf.mainopttstage (demand_df, 
                                                                                                       forecast_df, generators, batteries, instance_data, fisc_data,
                                                                                                       cost_data, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, solver_data, folder_path)
    
    else:   
        #Multiyear
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
        demand_df_i['demand'] = instance_data['demand_covered'] * demand_df_i['demand'] 
        
        #Calculate salvage, operation and replacement cost with investment cost
        generators, batteries = calculate_cost_data(generators, batteries, 
                                                    instance_data, cost_data)
    
    
        if (type_model == 'sm'):
            demand_df_year = copy.deepcopy(demand_df_i)
            forecast_df_year = copy.deepcopy(forecast_df_i)
            
            percent_df, energy_df, renew_df, total_df, brand_df, df_iterations, percent_df0, solutions = mf.mainstocmy(demand_df_year, 
                                                                                                                       forecast_df_year, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                                                       my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)
                
        else:
            demand_df, forecast_df = calculate_multiyear_data(demand_df_i, forecast_df_i,
                                                              my_data, instance_data['years'])  
    
    
            percent_df, energy_df, renew_df, total_df, brand_df  = mf.maindispatchmy(demand_df,
                                                                                     forecast_df, generators, batteries, instance_data, fisc_data, cost_data,
                                                                                     my_data, best_nsh, rand_ob, ADD_FUNCTION, REMOVE_FUNCTION, folder_path)



    return percent_df, energy_df, renew_df, total_df, brand_df
    


if __name__ == "__main__":
    main()
