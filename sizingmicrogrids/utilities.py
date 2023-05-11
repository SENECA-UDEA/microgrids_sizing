# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:51:07 2022

@author: scastellanos
"""

from sizingmicrogrids.classes import Solar, Eolic, Diesel, Battery
from sizingmicrogrids.strategies import dispatch_strategy, Results
from sizingmicrogrids.strategies import Results_my, dispatch_my_strategy
import pandas as pd
import requests
import json 
import numpy as np
import copy
import scipy.stats as st
import math
import datetime as dt



def read_data(demand_filepath, 
              forecast_filepath,
              units_filepath,
              instance_filepath,
              fiscal_filepath,
              cost_filepath):
    '''
    The codes uses the pd.read_csv() function from the pandas library to read 
    in the contents of the forecast_filepath and demand_filepath files 
    into dataframes forecast_df and demand_df respectively.

    It then uses the read_json_data function to read in the contents 
    of the units_filepath, instance_filepath, fiscal_filepath, 
    and cost_filepath into variables generators_data, instance_data, 
    fiscal_data, and cost_data respectively.

    Then it extracts the 'generators' and 'batteries' field from 
    the generators_data and assigns it to the variables
    generators and batteries respectively.

    Finally, the function returns the dataframes and variables
    Parameters
    ----------
    demand_filepath : PATH
        Demand data location
    forecast_filepath : PATH
        Forecast data Location (wind speed and irradiation)
    units_filepath : PATH
        Batteries and generators data location
    instance_filepath : PATH
        Instance paramaters data location
    fiscal_filepath : PATH
        Fiscal incentive data location
    cost_filepath : PATH
        Auxiliar cost data location - parameters for associated cost
        
    Returns
    -------
    demand_df, forecast_df : DATAFRAMES
    generators, batteries : LIST
    instance_data, fiscal_data, cost_data : DICTIONARIES
    '''
    forecast_df = pd.read_csv(forecast_filepath)
    demand_df = pd.read_csv(demand_filepath)

    generators_data = read_json_data(units_filepath)
    generators = generators_data.get('generators', {})
    batteries = generators_data.get('batteries', {})
    
    instance_data = read_json_data(instance_filepath)
    fiscal_data = read_json_data(fiscal_filepath)
    cost_data = read_json_data(cost_filepath)
    
    return demand_df, forecast_df, generators, batteries, instance_data, fiscal_data, cost_data


def read_json_data(filepath):
    '''
    This function takes in a file path as an argument. 
    The function first attempts to read the file path as a URL 
    and retrieves the data using the "requests" library and then 
    loads it as a JSON object. 
    If this fails (for example, if the file path is not a URL), 
    the code then attempts to open the file path as a local file 
    and load it as a JSON object using the "json" library. 
    If this also fails, the function will return an error. 

    Parameters
    ----------
    filepath : PATH
        Defines the location of the data to create a json

    Returns
    -------
    data : JSON
        Create a json data.

    '''
    try:
        data = requests.get(filepath)
        data = json.loads(data.text)
    except:
        with open(filepath) as f:
            data = json.load(f)
    return data


def create_objects(generators, batteries, forecast_df, 
                   demand_df, instance_data):
    '''
    This function creates objects for generators and batteries 
    based on their types (Solar, Eolic, Diesel, and Battery)
    and initializing them with certain values. 

    For each generator in the "generators" list, the function checks 
    the "tec" key to determine the type of generator. 
    If it is of type "S" (Solar), the function creates a Solar object,
    calls several methods on the object
    (irradiance_panel, get_inoct, solar_generation, solar_cost)
    and assigns the object to the key of "id_gen" in the "generators_dict" 
    If it is of type "W" (Eolic), the function creates an Eolic object, 
    calls several methods on the object (eolic_generation, eolic_cost) 
    and assigns the object to the key of "id_gen" in the "generators_dict" 
    .If it is of type "D" (Diesel) the function creates a Diesel object, 
    and assigns the object to the key of "id_gen" in the "generators_dict" 
   
    For each battery in the "batteries" list, the function creates
    a Battery object, calls the "calculate_soc" method on the object
    and assigns the object to the key of "id_bat" in the "batteries_dict" 

    Parameters
    ----------
    generators : LIST
    batteries : LIST
    forecast_df : DATAFRAME
    demand_df : DATAFRAME
    instance_data : DICTIONARY

    Returns
    -------
    generators_dict : DICTIONARY
        Dictionary with generators with their specific class - technology 
        associated
    batteries_dict : DICTIONARY
        Dictionary with batteries, and their attributes of battery class

    '''
    # Create generators and batteries
    generators_dict = {}
    for k in generators:
      if k['tec'] == 'S':
        obj_aux = Solar(*k.values())
        irr = irradiance_panel (forecast_df, instance_data)
        obj_aux.get_inoct(instance_data["caso"], instance_data["w"])
        obj_aux.solar_generation( forecast_df['t_ambt'], irr, instance_data["G_stc"], 0)
        obj_aux.solar_cost()
      elif k['tec'] == 'W':
        obj_aux = Eolic(*k.values())
        obj_aux.eolic_generation(forecast_df['Wt'], instance_data["h2"],
                                 instance_data["coef_hel"], 0)
        obj_aux.eolic_cost()
      elif k['tec'] == 'D':
        obj_aux = Diesel(*k.values())   
        
      generators_dict[k['id_gen']] = obj_aux
      
    batteries_dict = {}
    for l in batteries:
        obj_aux = Battery(*l.values())
        batteries_dict[l['id_bat']] = obj_aux
        batteries_dict[l['id_bat']].calculate_soc()
 
    return generators_dict, batteries_dict


def create_technologies(generators_dict, batteries_dict):
    '''
    This function creates two dictionaries of technologies and renewables based 
    on the generators and batteries dictionaries passed in as arguments.

    The "technologies_dict" is created by iterating through the generators_dict
    and batteries_dict values, checking if the technology attribute of the generator 
    or battery is already a key in the technologies_dict. If not, 
    it creates a new key with the "tec" attribute and adds the "br" attribute 
    to a set associated with that key. 
    
    The "renewables_dict" is created by iterating through the generators_dict,
    checking if the technology attribute of the generator is renewable,
    and if so, it checks if the "tec" attribute is already a key 
    in the renewables_dict. If not, it creates a new key with the "tec" attribute
    and adds the "br" attribute to a set associated with that key. 

    Parameters
    ----------
    generators_dict : DICTIONARY
    batteries_dict : DICTIONARY

    Returns
    -------
    technologies_dict : DICTIONARY
        dictionary of technologies present in generators,
        for example [Solar, Wind, Diesel and Batteries]
    renewables_dict : DICTIONARY
        dictionary that stores the technologies associated with generators 
        that are renewable such as solar and wind

    '''
    # Create technologies dictionary
    technologies_dict = dict()
    for bat in batteries_dict.values(): 
      if not (bat.tec in technologies_dict.keys()):
        technologies_dict[bat.tec] = set()
        technologies_dict[bat.tec].add(bat.br)
      else:
        technologies_dict[bat.tec].add(bat.br)

    for gen in generators_dict.values(): 
      if not (gen.tec in technologies_dict.keys()):
        technologies_dict[gen.tec] = set()
        technologies_dict[gen.tec].add(gen.br)
      else:
        technologies_dict[gen.tec].add(gen.br)

    # Creates renewables dict
    renewables_dict = dict()
    for gen in generators_dict.values(): 
        if gen.tec == 'S' or gen.tec == 'W': #or gen.tec = 'H'
          if not (gen.tec in renewables_dict.keys()):
              renewables_dict[gen.tec] = set()
              renewables_dict[gen.tec].add(gen.br)
          else:
              renewables_dict[gen.tec].add(gen.br)
              
    return technologies_dict, renewables_dict
 

def calculate_inverter_cost(generators_dict, batteries_dict, inverter_cost):
    '''
    This function calculates the total cost of inverters based on
    the generators and batteries information passed in as arguments.
    
    checking the technology attribute of each generator and adding the product
    of the generator's or batteries rated  power with the "inverter_cost" 

    Parameters
    ----------
    generators_dict : DICTIONARY
        DESCRIPTION.
    batteries_dict : DICTIONARY
        DESCRIPTION.
    inverter_cost : VALUE
        associated cost for each kilowatt of nominal capacity of the microgrid

    Returns
    -------
    expr : VALUE
        Total inverter cost

    '''
    expr = 0
    
    for gen in generators_dict.values(): 
        if (gen.tec == 'D'): 
            #cost as percentage of rated size network
            expr += gen.DG_max * inverter_cost
        elif (gen.tec == 'S'):
            #cost as percentage of rated size network
            expr += gen.Ppv_stc * inverter_cost                    
        else:
            #cost as percentage of rated size network
            expr += gen.P_y * inverter_cost

    for bat in batteries_dict.values(): 
        #cost as percentage of rated size network
        expr += bat.soc_max * inverter_cost

    return expr


#calculate total cost for two stage approach
def calculate_sizing_cost(generators_dict, batteries_dict, ir, years, 
                          delta, inverter):
    '''
    This function calculates the total cost of the two-stage approach based on
    the generators and batteries information passed in as arguments. 
    
    Then iterates through the generators_dict, checking the technology attribute 
    of each generator. If the generator is renewable, it adds the fiscal incentive
    to the cost.
    After, It then subtracts the generator's "cost_salvament".

    It then iterates through the batteries_dict and adds their cost
    
    Finally, the function calculates the cost of capital recovery factor (CRF) 
    using the formula: (ir * (1 + ir) ** (years)) / ((1 + ir) ** (years) - 1) 
    and multiplies the result with the total cost and returns the value.

    Parameters
    ----------
    generators_dict : DICTIONARY
    batteries_dict : DICTIONARY
    ir : VALUE
        Interest rate
    years : VALUE
        Sizing time horizon
    delta : VALUE
        discount rate as a tax incentive for the use of renewable energy
    inverter : VALUE
        Inverter cost

    Returns
    -------
    tnpccrf : VALUE
        Investment cost (long term - strategic) 
        together with the cost of operation make up the LCOE

    '''
    
    expr = 0
    
    for gen in generators_dict.values(): 
        if (gen.tec != 'D'): 
            #fiscal incentive if not diesel
            expr += gen.cost_up * delta
            expr += gen.cost_r  * delta
        else:
            expr += gen.cost_up
            expr += gen.cost_r 
        
        expr -= gen.cost_s 
        expr += gen.cost_fopm 
    
    for bat in batteries_dict.values(): 
        expr += bat.cost_up * delta
        expr += bat.cost_r * delta
        expr -= bat.cost_s
        expr += bat.cost_fopm

    crf = (ir * (1 + ir) ** (years)) / ((1 + ir) ** (years) - 1)    
    #Operative cost doesn't take into account the crf
    tnpccrf = (expr + inverter) * crf
    return tnpccrf


def calculate_area (solution):
    '''
    This function calculates the total area of the solution based on 
    the generators and batteries information passed in as argument "solution".}
    Then it iterates through the values of the dictionary with batteries and generators,
    and for each value, it adds the area attribute.

    Parameters
    ----------
    solution : OBJECT OF SOLUTION'S CLASS

    Returns
    -------
    area : Value
        Total area of installed microgrid.

    '''
    dict_actual = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
    area = 0
    for i in dict_actual.values():
        area += i.area 

    return area


#Calculate energy total, for every brand, technology or renewable 
def calculate_energy(batteries_dict, generators_dict, model_results, demand_df):  
    '''
    This function receives the sizing results and from them generates
    different dataframes that allow for subsequent analysis.

    Parameters
    ----------
    batteries_dict : DICTIONARY
    generators_dict : DICTIONARY
    model_results : OBJECT OF CLASS RESULTS
        The sizing outputs are stored here
    demand_df : DATAFRAME

    Returns
    -------
    percent_df : DATAFRAME
        percentage of demand covered by each generator or battery
    energy_df : DATAFRAME
        Power generared by each generator or battery
    renew_df : DATAFRAME
        Power generated by renewable generations
    total_df : DATAFRAME
        sum of total generation, at each period
    brand_df : DATAFRAME
        Power generated by each brand of generators or batteries

    '''
    #create auxiliar sets
    column_data = {}
    energy_data = {}
    aux_energy_data = []
    renew_data = {}
    aux_renew_data = []
    total_data = [0] * len(demand_df)
    aux_total_data = []
    brand_data = {}
    aux_brand_data = []
    for bat in batteries_dict.values(): 
        #check that the battery is installed
        if (model_results.descriptive['batteries'][bat.id_bat] == 1):
            column_data[bat.id_bat + '_%'] = (model_results.df_results[bat.id_bat + '_b-']
                                              / model_results.df_results['demand'])
            column_data[bat.id_bat + '_%charge'] = (model_results.df_results[bat.id_bat + '_b+']
                                                    / model_results.df_results['demand'])
            aux_total_data = model_results.df_results[bat.id_bat + '_b-']
            #sum all generation
            total_data += aux_total_data
            #check the key for create or continue in the same dict
            key_energy_total = bat.tec + 'total'
            key_brand_total = bat.br + 'total'
            if key_energy_total in energy_data:
                aux_energy_data = []
                aux_energy_data = (energy_data[key_energy_total] 
                                   + model_results.df_results[bat.id_bat + '_b-'])
                #fill the dictionary
                energy_data[key_energy_total] = aux_energy_data
            else:
                energy_data[key_energy_total] = model_results.df_results[bat.id_bat + '_b-']           
     
            if key_brand_total in brand_data:
                aux_brand_data = []
                aux_brand_data = (brand_data[key_brand_total] 
                                  + model_results.df_results[bat.id_bat + '_b-'])
                #fill the dictionary
                brand_data[key_brand_total] = aux_brand_data
            else:
                brand_data[key_brand_total] = model_results.df_results[bat.id_bat + '_b-']           
       
    for gen in generators_dict.values():
        #check that the generator is installed
        if (model_results.descriptive['generators'][gen.id_gen] == 1):
            column_data[gen.id_gen + '_%'] = (model_results.df_results[gen.id_gen] 
                                             / model_results.df_results['demand'])
            column_data[gen.id_gen + '_%'].where(column_data[gen.id_gen + '_%'] 
                                                 != np.inf, 0, inplace=True)
            #check the key for create or continue in the same dict
            key_energy_total = gen.tec + 'total'
            key_renew_total = gen.tec + 'total'
            key_brand_total = gen.br + 'total'
            #sum all generation
            total_data += model_results.df_results[gen.id_gen]
            if key_energy_total in energy_data:
                aux_energy_data = []
                aux_energy_data = (energy_data[key_energy_total] 
                                   + model_results.df_results[gen.id_gen])
                #fill the dictionary
                energy_data[key_energy_total] = aux_energy_data
            else:
                energy_data[key_energy_total] = model_results.df_results[gen.id_gen]           
            
            if key_brand_total in brand_data:
                aux_brand_data = []
                aux_brand_data = (brand_data[key_brand_total] 
                                  + model_results.df_results[gen.id_gen])
                #fill the dictionary
                brand_data[key_brand_total] = aux_brand_data
            else:
                brand_data[key_brand_total] = model_results.df_results[gen.id_gen]           
            
            if (gen.tec == 'S' or gen.tec == 'W'):
                if key_renew_total in renew_data:
                    aux_renew_data = []
                    aux_renew_data = (renew_data[key_renew_total] 
                                      +  model_results.df_results[gen.id_gen])
                    #fill the dictionary
                    renew_data[key_renew_total] = aux_renew_data
                else:
                    renew_data[key_renew_total] =  model_results.df_results[gen.id_gen]           
 
    #Create dataframes
    percent_df = pd.DataFrame(column_data, columns = [*column_data.keys()])
    energy_df = pd.DataFrame(energy_data, columns = [*energy_data.keys()])
    renew_df = pd.DataFrame(renew_data, columns = [*renew_data.keys()])
    arraydf = np.array(total_data)
    total_df = pd.DataFrame(arraydf, columns = ['Total energy'])
    brand_df = pd.DataFrame(brand_data, columns = [*brand_data.keys()])
    
    return percent_df, energy_df, renew_df, total_df, brand_df


def calculate_percent_tec (sol_best, percent_df):
    """
    This function provides the average percentage that each technology 
    provides to the microgrid

    Parameters
    ----------
    sol_best : OBJECT OF SOLUTION CLASS
    percent_df : DATAFAME

    Returns
    -------
    mean_total_d : PERCENT
        Average percentage of load satisfied by diesel generation
    mean_total_s : PERCENT
        Average percentage of load satisfied by solar generation
    mean_total_w : PERCENT
        Average percentage of load satisfied by eolic generation
    mean_total_b : PERCENT
        Average percentage of load satisfied by diesel generation

    """
    list_d, list_s, list_w, list_b = [], [], [], []
    for i in sol_best.generators_dict_sol.values():
        if i.tec == 'S':
            list_s.append(i.id_gen)
        elif i.tec == 'D':
            list_d.append(i.id_gen)
        elif i.tec == 'W':
            list_w.append(i.id_gen)
    for i in sol_best.batteries_dict_sol.values():
        list_b.append(i.id_bat)
    
    mean_total_d = sum(percent_df[f"{i}_%"].mean() for i in list_d)
    mean_total_s = sum(percent_df[f"{i}_%"].mean() for i in list_s)
    mean_total_w = sum(percent_df[f"{i}_%"].mean() for i in list_w)
    mean_total_b = sum(percent_df[f"{i}_%"].mean() for i in list_b if f"{i}_%" in percent_df)

    return mean_total_d, mean_total_s, mean_total_w, mean_total_b


def create_excel(sol_best, percent_df, name_file, folder_file, lcoe_scn = 0, robust_scn = 0, type_model = 0):
    """
    This function creates a excel file with the most relevant results
    
    Parameters
    ----------
    sol_best : SOLUTION OBJECT.
    percent_df : DATAFRAME
    name_file : STRING
    folder_file STRING
        Location to save the file
    lcoe_scn : VALUE, optional
        Average best scenario lcoe
    robust_scn : VALUE, optional
        Average robustness best scenario
    type_model: BINARY, default = 0
        0 = deterministic
        1 = stochastic
    """
    lcoe = sol_best.results.descriptive['LCOE']
    area = sol_best.results.descriptive['area']
    lpsp_mean = sol_best.results.df_results['LPSP'].mean()
    wasted_mean = sol_best.results.df_results['Wasted Energy'].mean()
    mean_total_d, mean_total_s, mean_total_w, mean_total_b = calculate_percent_tec (sol_best,
                                                                                  percent_df)
    dict_tecs = {}
    generators = sol_best.generators_dict_sol.items()
    batteries = sol_best.batteries_dict_sol.items()
    
    for id_gen, generator in generators:
        dict_tecs[generator.id_gen] = generator.tec
    
    for id_bat, battery in batteries:
        dict_tecs[battery.id_bat] = battery.tec
    
    pd_tecs = pd.DataFrame.from_dict(dict_tecs, orient='index', columns=["tec"])
    
    if (type_model == 0):
        rows_df_results = [[lcoe, area, lpsp_mean, wasted_mean,
                                mean_total_d, mean_total_w, mean_total_s,
                                mean_total_b]]
        
        results_report = pd.DataFrame(rows_df_results, columns=[ "Lcoe","area","lpsp mean",
                                                                "mean surplus","mean diesel generation", "mean eolic generation",
                                                                "mean solar generation","mean batteries generation"])
    elif (type_model == 1):
        rows_df_results = [[lcoe, area, lpsp_mean, wasted_mean,
                                mean_total_d, mean_total_w, mean_total_s,
                                mean_total_b, lcoe_scn, robust_scn]]
        
        results_report = pd.DataFrame(rows_df_results, columns=[ "Lcoe","area","lpsp mean",
                                                                "mean surplus","mean diesel generation", "mean eolic generation",
                                                                "mean solar generation","mean batteries generation",
                                                                "average lcoe best solution", "robustness best solution"])

    
    results_report = results_report.T     
    results_report.rename(columns = {0:'Results'}, inplace = True)
    name_excel = str(name_file) + '.xlsx' 
    #route_excel = "../data/Results_output/" + name_excel
    route_excel = str(folder_file) + "/" + name_excel
    
    writer = pd.ExcelWriter(route_excel, engine='xlsxwriter')
    results_report.to_excel(writer, sheet_name='descriptive results', startrow = 0 , startcol = 0)
    pd_tecs.to_excel(writer, sheet_name='descriptive results', startrow = 0 , startcol = 3)
    workbook  = writer.book
    worksheet = writer.sheets['descriptive results']
    
    #percent format
    percent_format = workbook.add_format({'num_format': '0.00%'})    
    
    # Format
    worksheet.write('B4', lpsp_mean, percent_format)
    worksheet.write('B5', wasted_mean, percent_format)
    worksheet.write('B6', mean_total_d, percent_format)
    worksheet.write('B7', mean_total_w, percent_format)
    worksheet.write('B8', mean_total_s, percent_format)
    worksheet.write('B9', mean_total_b, percent_format)
    
    #results sheet
    sol_best.results.df_results.to_excel(writer, sheet_name='Results')
    worksheet = writer.sheets['Results']
    column_index = sol_best.results.df_results.columns.get_loc('S-')
    worksheet.set_column(column_index + 2, column_index + 2, None, percent_format)    
    
    #percent sheet
    percent_df.to_excel(writer, sheet_name='Percent')    
    worksheet = writer.sheets['Percent']
    column_index = len(percent_df)
    worksheet.set_column(1, column_index + 1, None, percent_format)
        
    writer.close()  

    return 
    
def interest_rate (i_f, inf):
    '''
    calculates interest rate according to the relationship nominal rate and inflation,
    formula used in financial engineering
    
    Parameters
    ----------
    i_f : VALUE
        Nominal rate
    inf : VALUE
        Inflation

    Returns
    -------
    ir : Value
        Interest rate

    '''

    ir = (i_f - inf) / (1 + inf)
    return ir


def calculate_cost_data(generators, batteries, instance_data,
                        parameters_cost):
    '''
    This function calculates the costs associated with a given set 
    of generators and batteries over a specified number of years. 
    It first calculates the interest rate based on the given inflation and nominal rate. 
    Then, it calculates the tax for replacement based on the number of cycles 
    and the useful life of the generators and batteries. 
    
    The function then iterates through the generators and batteries,
    calculating costs associated with each one, such as replacement cost,
    salvage cost, fixed and variable operation and maintenance costs. 
    These costs are stored in new lists and returned at the end of the function.

    Parameters
    ----------
    generators : DICTIONARY
    batteries : DICTIONARY
    instance_data : DICTIONARY
    parameters_cost : DICTIONARY

    Returns
    -------
    generators_def : DICTIONARY
    batteries_def : DICTIONARY

    '''
    #inflation
    inf = instance_data['inf']
    #nominal rate
    i_f = instance_data['i_f']
    years = instance_data['years']
    ir = interest_rate(i_f, inf)
    #defaul useful life Diesel and batteries = 10
    life_cicle = parameters_cost['life_cicle']
    n_cycles = years / life_cicle
    
    #Calculate tax for remplacement
    tax = 0
    for h in range(1, int(n_cycles) + 1):
        tax += 1 / ((1 + ir)**(h * life_cicle))
        
    aux_generators = []
    aux_batteries = []
    #definitive list
    generators_def = []
    batteries_def = []
    #Calculate costs with investment cost
    for i in generators:
        if (i['tec'] == 'S'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = parameters_cost['param_r_solar']
            aux_generators['cost_s'] = (cost_up * parameters_cost['param_s_solar'] 
                                        * (((1 + inf) / (1 + ir)) ** years))
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_solar'] 
            aux_generators['cost_vopm'] = cost_up * parameters_cost['param_v_solar']      
        elif (i['tec'] == 'W'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = parameters_cost['param_r_wind']  
            aux_generators['cost_s'] = (cost_up * parameters_cost['param_s_wind'] 
                                        * (((1 + inf) / (1 + ir))**years))
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_wind']  
            aux_generators['cost_vopm'] = cost_up * parameters_cost['param_v_wind']              
        elif (i['tec'] == 'D'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = cost_up * parameters_cost['param_r_diesel'] * tax
            aux_generators['cost_s'] = (cost_up * parameters_cost['param_s_diesel']   
                                        * (((1 + inf) / (1 + ir))**years))
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_diesel']
        generators_def.append(copy.deepcopy(aux_generators))  
        
    for i in batteries:
        cost_up = i['cost_up']
        aux_batteries = []
        aux_batteries = i
        aux_batteries['cost_r'] = cost_up * parameters_cost['param_r_bat'] * tax
        aux_batteries['cost_s'] = (cost_up * parameters_cost['param_s_bat']
                                   * (((1 + inf)/(1 + ir))**years))
        aux_batteries['cost_fopm'] = cost_up * parameters_cost['param_f_bat']
        aux_batteries['cost_vopm'] = cost_up * parameters_cost['param_v_bat']
        batteries_def.append(copy.deepcopy(aux_batteries))

    return generators_def, batteries_def


def fiscal_incentive (credit, depreciation, corporate_tax, ir, T1, T2):
    '''
    This function calculates the fiscal incentives associated with a 
    power generating facility. It takes in five parameters: 
    credit, depreciation, corporate tax rate, interest rate, and T1 and T2. 
    It calculates the tax savings over T1 years by applying the investment tax credit,
    and over T2 years by applying the depreciation. 
    
    It then calculates the overall tax savings using the corporate tax rate 
    and the savings from the credit and depreciation. The final result is returned as delta.

    Parameters
    ----------
    credit : VALUE
        Investment tax credit
    depreciation : VALUE
        Factor expressed as % of investment cost over T2 year
    corporate_tax : VALUE
        Effective corporate tax income rate
    ir : VALUE
        Interest rate
    T1 : VALUE
        Maximum number of years to apply the investment tax credit
    T2 : VALUE
        Useful life (year) of the power generating facility - depreciation

    Returns
    -------
    delta : VALUE
        Tax incentive: discount percentage to the investment cost of renewable energies.

    '''

    delta = 0
    expr = 0
    for j in range(1, int(T1) + 1):
        expr += credit / ((1 + ir)**j)

    for j in range(1, int(T2) + 1):
        expr += depreciation / ((1 + ir)**j)
    
    delta = (1 / (1 - corporate_tax)) * (1 - corporate_tax * expr)
    
    return delta


'''ILS'''

def ils(N_ITERATIONS, sol_current, sol_best, search_operator,
        REMOVE_FUNCTION, ADD_FUNCTION,delta, rand_ob, instance_data, AMAX,
        demand_df, cost_data, type_model, best_nsh, CRF = 0, ir = 0, my_data = {}):
    '''
    This function performs the iterated local search, creates a dataframe 
    to save the report of each iteration, part of a current solution and a 
    better solution, which at the beginning are the same, depending on the
    solution that is handling whether it is feasible or not, determines
    whether to do the function add or the function remove, it is also checked
    if the ils have to do grasp or random. As there are different models, 
    the model has slight variations if it is multi-year or not.
    If the solution is feasible it saves it and generates the results,
    and if it has the lowest lcoe it saves it as the best solution and counts
    the number of hours not served for this solution.
    
    At the end it calculates the area of the solution to check
    in the next iteration if there is more space or not and at the end
    it has the report of the best solution together with its results.

    Parameters
    ----------
    N_ITERATIONS : NUMBER - INTEGER
    sol_current : OBJECT OF SOLUTION CLASS
    sol_best : OBJECT OF SOLUTION CLASS
    search_operator : OBJECT OF OPERATORS
    REMOVE_FUNCTION : STRING
        GRASP or RANDOM
    ADD_FUNCTION : STRING
        GRASP or RANDOM
    delta : NUMBER - PERCENT
    rand_ob : OBJECT OF RANDOM CLASS
    instance_data : NUMBER - PERCENT
    AMAX : NUMBER - DF
    demand_df : DATAFRAME
    cost_data : DICTIONARY
    type_model : String
        ILS-DS = Iterated local search + Dispatch Strategy
        ILS-DS-MY = Iterated local search + Dispatch Strategy + Multiyear
    best_nsh : VALUE - INTEGER
        initial best not served hours of initial best solution
    CRF : NUMBER - PERCENT, optional
        Used only in not multiyear model. The default is 0, to avoid errors
    ir : NUMBER - PERCENT, optional
        Used only in multiyear model. The default is 0, to avoid errors
    my_data : DICTIONATY, optional
        Used only in multiyear model.. The default is {}, to avoid errors

    Returns
    -------
    sol_best : OBJECT OF SOLUTION CLASS
        Best solution that solves the model - lowest LCOE
    best_nsh : NUMBER - INTEGER
        Number of not served hours in the best solution
    rows_df : DATAFRAME
        Iterations report

    '''

    rows_df = []
    movement = "initial solution"
    for i in range(N_ITERATIONS):
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
                if (type_model == 'ILS-DS'):                       
                    sol_try, remove_report = search_operator.remove_object(sol_current, 
                                                                           delta, rand_ob, 'DS', CRF)
                elif (type_model == 'ILS-DS-MY'):
                    sol_try, remove_report = search_operator.remove_object(sol_current, delta, 
                                                                           rand_ob, 'MY', 0)
            elif (REMOVE_FUNCTION == 'RANDOM'):
                sol_try, remove_report = search_operator.remove_random_object(sol_current, rand_ob)
    
            movement = "Remove"
        else:
            #  Create list of generators that could be added
            list_available_bat, list_available_gen, list_tec_gen  = search_operator.available_items(sol_current, AMAX)
            if (list_available_gen != [] or list_available_bat != []):
                # Add a generator or battery to the current solution
                if (ADD_FUNCTION == 'GRASP'):
                    if (type_model == 'ILS-DS'): 
                        sol_try, remove_report = search_operator.add_object(sol_current, 
                                                                            list_available_bat, list_available_gen, list_tec_gen, remove_report, 
                                                                            instance_data['fuel_cost'], rand_ob, delta, 'DS', CRF)
                    
                    elif (type_model == 'ILS-DS-MY'):
                        sol_try, remove_report = search_operator.add_object(sol_current, 
                                                                            list_available_bat, list_available_gen, list_tec_gen, 
                                                                            remove_report, instance_data['fuel_cost'], rand_ob, delta, 'MY', 0)
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
        #instance_data['inverter cost'] = calculate_inverter_cost(sol_try.generators_dict_sol,
        #sol_try.batteries_dict_sol,val)
        
        #Run the dispatch strategy process
        if (type_model == 'ILS-DS'):
            lcoe_cost, df_results, state, time_f, nsh = dispatch_strategy(sol_try, demand_df,
                                                                          instance_data, cost_data, CRF, delta, rand_ob)
        elif (type_model == 'ILS-DS-MY'):
            lcoe_cost, df_results, state, time_f, nsh = dispatch_my_strategy(sol_try, demand_df, 
                                                                             instance_data, cost_data, delta, rand_ob, my_data, ir)
        #print("finish simulation - state: " + state)
        #Create results
        if state == 'optimal':
            if (type_model == 'ILS-DS'):
                sol_try.results = Results(sol_try, df_results, lcoe_cost)
            elif (type_model == 'ILS-DS-MY'):
                sol_try.results = Results_my(sol_try, df_results, lcoe_cost)
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
            if (type_model == 'ILS-DS'):
                sol_try.results = Results(sol_try, df_results, lcoe_cost)
            elif (type_model == 'ILS-DS-MY'):
                sol_try.results = Results_my(sol_try, df_results, lcoe_cost)
            sol_current = copy.deepcopy(sol_try)
        
        #calculate area
        sol_current.results.descriptive['area'] = calculate_area(sol_current)
        #delete to avoid overwriting
        del df_results
        del sol_try
        
    return sol_best, best_nsh, rows_df    
      
'MULTIYEAR'

def read_multiyear_data(demand_filepath, 
                        forecast_filepath,
                        units_filepath,
                        instance_filepath,
                        fiscal_filepath,
                        cost_filepath,
                        my_filepath):
    '''
    The codes uses the pd.read_csv() function from the pandas library to read 
    in the contents of the forecast_filepath and demand_filepath files 
    into dataframes forecast_df and demand_df respectively.

    It then uses the read_json_data function to read in the contents 
    of the units_filepath, instance_filepath, fiscal_filepath, my_filepath 
    and cost_filepath into variables generators_data, instance_data, 
    fiscal_data, my_data and cost_data respectively.

    Then it extracts the 'generators' and 'batteries' field from 
    the generators_data and assigns it to the variables
    generators and batteries respectively.

    Finally, the function returns the dataframes and variables
    
    Unlike its similar function, this one has an additional variable
    as it is used for the multi-year project.

    Parameters
    ----------
    demand_filepath : PATH
        Demand data location
    forecast_filepath : PATH
        Forecast data Location (wind speed and irradiation)
    units_filepath : PATH
        Batteries and generators data location
    instance_filepath : PATH
        Instance paramaters data location
    fiscal_filepath : PATH
        Fiscal incentive data location
    cost_filepath : PATH
        Auxiliar cost data location - parameters for associated cost
    my_filepath : PATH
        Data location for multi-year calculations
        
    Returns
    -------
    demand_df, forecast_df : DATAFRAMES
    generators, batteries : LIST
    instance_data, fiscal_data, cost_data, my_data : DICTIONARIES

    '''
    forecast_df = pd.read_csv(forecast_filepath)
    demand_df = pd.read_csv(demand_filepath)

    generators_data = read_json_data(units_filepath)
    generators = generators_data.get('generators', {})
    batteries = generators_data.get('batteries', {})
    
    instance_data = read_json_data(instance_filepath)
    fiscal_data = read_json_data(fiscal_filepath)
    cost_data = read_json_data(cost_filepath)
    my_data = read_json_data(my_filepath)
   
    return demand_df, forecast_df, generators, batteries, instance_data, fiscal_data, cost_data, my_data


def create_multiyear_objects(generators, batteries, forecast_df, 
                             demand_df, instance_data, my_data):
    
    '''
    This function creates objects for generators and batteries 
    based on their types (Solar, Eolic, Diesel, and Battery)
    and initializing them with certain values. 

    For each generator in the "generators" list, the function checks 
    the "tec" key to determine the type of generator. 
    If it is of type "S" (Solar), the function creates a Solar object,
    calls several methods on the object
    (irradiance_panel, get_inoct, solar_generation, solar_cost)
    and assigns the object to the key of "id_gen" in the "generators_dict" 
    If it is of type "W" (Eolic), the function creates an Eolic object, 
    calls several methods on the object (eolic_generation, eolic_cost) 
    and assigns the object to the key of "id_gen" in the "generators_dict" 
    .If it is of type "D" (Diesel) the function creates a Diesel object, 
    and assigns the object to the key of "id_gen" in the "generators_dict" 
   
    For each battery in the "batteries" list, the function creates
    a Battery object, calls the "calculate_soc" method on the object
    and assigns the object to the key of "id_bat" in the "batteries_dict" 
   
    Unlike its similar function, this one has an additional variable
    as it is used for the multi-year project; 
    so use equipment degradation rate for each year
    
    Parameters
    ----------
    generators : LIST
    batteries : LIST
    forecast_df : DATAFRAME
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    my_data : DICTIONARY

    Returns
    -------
    generators_dict : DICTIONARY
        Dictionary with generators with their specific class - technology 
        associated
    batteries_dict : DICTIONARY
        Dictionary with batteries, and their attributes of battery class

    '''
    generators_dict = {}
    for k in generators:
      if k['tec'] == 'S':
        obj_aux = Solar(*k.values())
        irr = irradiance_panel (forecast_df, instance_data)
        obj_aux.get_inoct(instance_data["caso"], instance_data["w"])
        obj_aux.solar_generation( forecast_df['t_ambt'], irr,
                                 instance_data["G_stc"], my_data["sol_deg"])
        obj_aux.solar_cost()
      elif k['tec'] == 'W':
        obj_aux = Eolic(*k.values())
        obj_aux.eolic_generation(forecast_df['Wt'], instance_data["h2"]
                                 , instance_data["coef_hel"], my_data["wind_deg"] )
        obj_aux.eolic_cost()
      elif k['tec'] == 'D':
        obj_aux = Diesel(*k.values())   
        
      generators_dict[k['id_gen']] = obj_aux
      
    batteries_dict = {}
    for l in batteries:
        obj_aux = Battery(*l.values())
        batteries_dict[l['id_bat']] = obj_aux
        batteries_dict[l['id_bat']].calculate_soc()
    return generators_dict, batteries_dict


def calculate_multiyear_data(demand_df, forecast_df, my_data, years):
    '''
    It creates two new dataframes called 'demand' and 'forecast'
    that are multi-year dataframes based on the input dataframes. 
    
    It first creates empty dataframes with the same columns as the input
    dataframes and the same number of rows as the number of years passed 
    in multiplied by 8760 (number of hours in a year). It then populates
    these dataframes with the data from the input dataframes,
    adjusting values as necessary (e.g. applying a demand tax) 
    for each year beyond the first year. 
    
    It has the binary option of using the default data or not,
    If 1 is placed, the data entered by the user that already contains 
    all the years is used
    If 0 is placed, the program makes the projection 
    from the first year to the other years.
    
    Example default data
    --------------------
            1 = do not do calculations, the user already has the entire time horizon
            0 = do the calculations as a first year projection
    
    Parameters
    ----------
    demand_df : DATAFRAME
    forecast_df : DATAFRAME
    my_data : DICTIONARY
    years : PARAMETER
        Project time horizon

    Returns
    -------
    demand : DATAFRAME
        Multiyear demand
    forecast : DATAFRAME
        Multiyear forecast

    '''
    #check default data
    default_data = my_data['default_data']
    if (default_data == 1):
        demand = copy.deepcopy(demand_df)
        forecast = copy.deepcopy(forecast_df)
    else:
        #total hours
        len_total = 8760 * years
        aux_demand = {k : [0] * (len_total) for k in demand_df}
        aux_forecast = {k : [0] * (len_total) for k in forecast_df}
        
        for i in range(len_total):
            aux_demand['t'][i] = i
            aux_forecast['t'][i] = i
            #first year same 
            if (i < 8760):    
                aux_demand['demand'][i] = demand_df['demand'][i]
                aux_forecast['DNI'][i] = forecast_df['DNI'][i]
                aux_forecast['t_ambt'][i] = forecast_df['t_ambt'][i]
                aux_forecast['Wt'][i] = forecast_df['Wt'][i]
                aux_forecast['Qt'][i] = forecast_df['Qt'][i]
                aux_forecast['GHI'][i] = forecast_df['GHI'][i]
                aux_forecast['day'][i] = forecast_df['day'][i]
                aux_forecast['SF'][i] = forecast_df['SF'][i]
                aux_forecast['DHI'][i] = forecast_df['DHI'][i]
            #others years
            else:
                #get the year
                k = math.floor(i / 8760)
                #apply tax
                val = demand_df['demand'][i - 8760 * k] * (1 + my_data["demand_tax"]) ** k
                #asign value
                aux_demand['demand'][i] = val
                #forecast is the same that first year
                val2 = forecast_df['DNI'][i - 8760 * k]
                aux_forecast['DNI'][i] = val2
                aux_forecast['t_ambt'][i] = forecast_df['t_ambt'][i - 8760 * k]
                val3 = forecast_df['Wt'][i - 8760 * k]
                aux_forecast['Wt'][i] = val3
                aux_forecast['Qt'][i] = forecast_df['Qt'][i - 8760 * k]
                val4 = forecast_df['GHI'][i - 8760 * k]
                aux_forecast['GHI'][i] = val4
                aux_forecast['day'][i] = forecast_df['day'][i - 8760 * k]
                aux_forecast['SF'][i] = forecast_df['SF'][i - 8760 * k]
                val5 = forecast_df['DHI'][i - 8760 * k]
                aux_forecast['DHI'][i] = val5      
                
        #create dataframe
        demand = pd.DataFrame(aux_demand, columns=['t','demand'])
        forecast = pd.DataFrame(aux_forecast, columns=['t','DNI','t_ambt','Wt', 
                                               'Qt','GHI','day','SF','DHI'])
        
    return demand, forecast

'STOCHASTICITY'

#create the hourly dataframe
def hour_data(data):
    '''
    This function takes in a dataframe as input, and creates a new dataframe
    with hourly granularity. 
    
    The new dataframe is represented as a dictionary, 
    where the keys are the hours of the day (0-23) and the values are lists 
    of data for that hour, one element for each day. The original dataframe's 
    index is used to determine the hour and day for each data point, 
    which is then used to populate the appropriate element in the new dataframe. 

    Parameters
    ----------
    data : DICTIONARY
        Year data used

    Returns
    -------
    vec : DATAFRAME
        new hourly dataframe.

    '''
    hours_size = len(data) / 24
    vec = {k : [0] * int(hours_size) for k in range(int(24))}
    
    for t in data.index.tolist():
        #get the hour
        hour_day = t % 24
        #get the day
        day_year = math.floor(t / 24)
        #create data
        vec[hour_day][day_year] = data[t]
        
    return vec


def week_vector_data(data, year, first_day=1):
    """
    This function separates data into separate dataframes
    for weekdays and weekends. The input "data" is a dataframe containing data 
    that the user wants to separate by weekdays and weekends. 
    The input "year" is an integer specifying the year the data is from, 
    and "first_day" is an integer specifying the first day of the data,
    with a default value of 1 (January 1st). The function then calculates
    the initial and final days of the data, and uses the number of weekdays
    and weekends to create two dataframes "dem_week_vec" and "dem_weekend_vec"
    with the data separated by weekday and weekend respectively.
    The function then returns the two dataframes.

    Parameters
    ----------
    data : DATAFRAME
        Data from which the time separation between week and weekend 
        will be made, for example the demand.
    year : INTEGER
        year from which the data is taken, for example 2018
    first_day : INTEGER
        first day of the database, by default it starts on the first of January, 
        a number between 1 and 365 is placed, 
        the number 32 would correspond to the first of February

    Returns
    -------
    dem_week_vec : DATAFRAME
        Hourly Dataframe, and for each day fill it with the demand data 
        that is not a weekend
    dem_weekend_vec : DATAFRAME
        Hourly Dataframe, and for each day fill it with the demand data 
        that is a weekend

    """
    #first day of the year
    first_january = dt.date(year, 1, 1)
    #check the day of the fist day of the data
    initial_day = dt.timedelta(int(first_day) - 1) + first_january
    initial_day_number = int(initial_day.strftime("%w"))
    #get the last day
    hours_size = len(data) / 24
    final_day = dt.timedelta(int(hours_size)) + initial_day
    
    #number of week and weekend days
    day_number = initial_day_number
    end = final_day
    start = initial_day
    days = np.busday_count(start, end)

    dem_week_vec = {k : [0] * int(days) for k in range(int(24))}
    dem_weekend_vec = {k : [0] * int(hours_size - days) for k in range(int(24))}
    day_week = -1
    day_weekend = -1
    
    for t in data.index.tolist():
        #get the hour
        hour_day = t%24
        #calculate the week day
        if ((hour_day == 0) and (day_number == 6) and (t != 0)):
            day_number = 0
        elif ((hour_day == 0) and (t != 0)):
            day_number += 1
        
        
        if (day_number != 6 and day_number != 0):
            #create data
            dem_week_vec[hour_day][day_week] = data[t]
            if (hour_day == 0):
                day_week += 1
        else:
            dem_weekend_vec[hour_day][day_weekend] = data[t]
            if (hour_day == 0):
                day_weekend += 1
    return dem_week_vec, dem_weekend_vec


def get_best_distribution(vec):
    '''
    Fix a distribution for each set

    Parameters
    ----------
    vec : DATAFRAME

    Returns
    -------
    dist : DICTIONARY
        To each vector in the data frame returns 
        the best associated probability distribution

    '''
    #get the total hours
    hours = len(vec)
    dist = {}

    #calculate distribution of df
    for i in range(int(hours)):
        dist[i] = best_distribution(vec[i])
    return dist


def best_distribution(data):
    '''
    This function takes in a data set as input, and attempts to find
    the best fitting probability distribution for the data. 
    It does this by iterating through a list of pre-defined distributions 
    using the SciPy library's "getattr" function and "fit" method. 
    It then applies the Kolmogorov-Smirnov test to each distribution
    and gets the p-value. 
    The function then selects the distribution that has the highest p-value 
    and returns the name of the distribution, its p-value and its parameters.
    If the sum of the data is 0 and the standard deviation is also 0,
    the function returns "No distribution" and None as the best distribution.

    Parameters
    ----------
    data : DATAFRAME

    Returns
    -------
    best_dist : VALUE
    best_p : VALUE
    params[best_dist] : VALUE
        Parameters associated to the best distribution

    '''
    #available distributions
    dist_names = [
        "norm","weibull_max","weibull_min","pareto", 
        "gamma","beta","rayleigh","invgauss",
        "uniform","expon", "lognorm","pearson3","triang"
        ]
    dist_results = []
    params = {}
    #if 0 no distribution, 0 value, example solar generator at night
    if (sum(data) == 0 and np.std(data) == 0):
        best_dist = 'No distribution'
        best_p = None
        params[best_dist] = 0
    else:
        #fit each distribution
        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test and get p-value
            D, p = st.kstest(data, dist_name, args = param)
            dist_results.append((dist_name, p))
    
        # select the best fitted distribution
        best_dist, best_p = (max(dist_results, key = lambda item: item[1]))
        # store the name of the best fit and its p value

    return best_dist, best_p, params[best_dist]


#create stochastic df
def calculate_stochasticity_demand(rand_ob, demand_df_i, week_dist, weekend_dist,
                                   year, first_day=1):
    '''
    The function iterates over the time index of the demand_df dataframe.
    For each time step, it extracts the hour of the day from the index and
    uses it to look up the appropriate probability distribution from 
    the dem_dist dictionary. Previously, the model checks whether it should use
    the weekly or weekend distribution, depending on the corresponding day
    of the week. It then generates a random number using
    the rand_ob object and the selected probability distribution. 
    The function then updates the demand_df dataframe by adding 
    the generated random number as a new column at the corresponding time step.
    Finally, the function returns the updated demand_df dataframe.

    Parameters
    ----------
    rand_ob : OBJECT OF RANDOM GENERATOR CLASS
        Function to calculate random values or sets
    demand_df_i : DATAFRAME
    week_dist : DICTIONARY
    weekend_dist : DICTIONARY
    Returns
    year : INTEGER
        year from which the data is taken, for example 2018
    first_day : INTEGER
        first day of the database, by default it starts on the first of January, 
        a number between 1 and 365 is placed, 
        the number 32 would correspond to the first of February
    -------
    demand_df : DATAFRAME
        New dataframe

    '''
    demand_df = copy.deepcopy(demand_df_i)
    #first day of the year
    first_january = dt.date(year, 1, 1)
    #check the day of the fist day of the data
    initial_day = dt.timedelta(int(first_day) - 1) + first_january
    initial_day_number = int(initial_day.strftime("%w"))
    
    #number of week and weekend days
    day_number = initial_day_number
              
    for t in demand_df['t']:
        #get the hour for the distribution
        hour = t % 24
        #calculate the week day
        if ((hour == 0) and (day_number == 6) and (t != 0)):
            day_number = 0
        elif ((hour == 0) and (t != 0)):
            day_number += 1
        #generate one random number for each hour, select week or weekend
        if (day_number != 6 and day_number != 0):  
            obj = generate_random(rand_ob, week_dist[hour])
            demand_df.loc[t] = [t,obj]
        else:
            obj = generate_random(rand_ob, weekend_dist[hour])
            demand_df.loc[t] = [t,obj]            
  
    return demand_df
    

def calculate_stochasticity_forecast(rand_ob, forecast_df_i, wind_dist,
                                     sol_distdni, sol_distdhi, sol_distghi):
    '''
    The function iterates over the time index of the forecast_df dataframe.
    For each time step, it extracts the hour of the day from the index 
    and uses it to look up the appropriate probability distributions 
    from the wind_dist, sol_distdni, sol_distdhi, and sol_distghi dictionaries.
    It then generates a random number using the rand_ob object 
    and the selected probability distributions. 
    The function then updates the forecast_df dataframe by adding 
    the generated random numbers as new columns at the corresponding time step. 

    Parameters
    ----------
    rand_ob : OBJECT OF RANDOM GENERATOR CLASS
        Function to calculate random values or sets
    forecast_df : DATAFRAME
    wind_dist : DICTIONARY
        For Wind speed.
    sol_distdni : DICTIONARY
        For Direct normal Irradiance
    sol_distdhi : DICTIONARY
        For Diffuse Horizontal Irradiance
    sol_distghi : DICTIONARY
        For Global horizontal Irradiance

    Returns
    -------
    forecast_df : DATAFRAME
        New dataframe

    '''
    forecast_df = copy.deepcopy(forecast_df_i)
    for t in forecast_df['t']:
        #get the hour for the distribution
        hour = t % 24
        #generate one random number for each hour (demand and forecast)
        nf_w = generate_random(rand_ob, wind_dist[hour])
        nf_dni = generate_random(rand_ob, sol_distdni[hour])
        nf_dhi = generate_random(rand_ob, sol_distdhi[hour])
        nf_ghi = generate_random(rand_ob, sol_distghi[hour])
        t_ambt = forecast_df['t_ambt'][t]
        Qt = forecast_df['Qt'][t]
        day = forecast_df['day'][t]
        SF = forecast_df['SF'][t]
        forecast_df.loc[t] = [t, nf_dni, t_ambt, nf_w, Qt, nf_ghi, day, SF, nf_dhi]
               
    return forecast_df
    

#generate one random number with distribution
def generate_random(rand_ob, dist):
    '''
    The function first checks the name of the distribution from dist[0], 
    It then checks different probability distributions and call 
    the appropriate method of rand_ob and pass the parameters if the name
    of the distribution is matched.
    
    If the distribution name is not matched with any of the names it returns 0.
    Finally, the function returns the generated random number.

    Parameters
    ----------
    rand_ob : TYPE
        DESCRIPTION.
    dist : DICTIONARY
        Contains the name of the distribution and its parameters

    Returns
    -------
    Number : VALUE
        Random number generated

    '''
    if (dist[0] == 'norm'):
        number = rand_ob.dist_norm(dist[2][0], dist[2][1])
    elif (dist[0] == 'uniform'):
        number = rand_ob.dist_uniform(dist[2][0], dist[2][1])
    elif(dist[0] == 'No distribution'):
        number = 0
    elif (dist[0] == 'triang'):
        number = rand_ob.dist_triang(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'weibull_max'):
        number = rand_ob.dist_weibull_max(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'weibull_min'):
        number = rand_ob.dist_weibull_min(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'pareto'):
        number = rand_ob.dist_pareto(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'gamma'):
        number = rand_ob.dist_gamma(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'beta'):
        number = rand_ob.dist_beta(dist[2][0], dist[2][1], dist[2][2], dist[2][3])
    elif (dist[0] == 'rayleigh'):
        number = rand_ob.dist_rayleigh(dist[2][0], dist[2][1])
    elif (dist[0] == 'invgauss'):
        number = rand_ob.dist_invgauss(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'expon'):
        number = rand_ob.dist_expon(dist[2][0], dist[2][1])
    elif (dist[0] == 'lognorm'):
        number = rand_ob.dist_lognorm(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'pearson3'):
        number = rand_ob.dist_pearson3(dist[2][0], dist[2][1], dist[2][2])        
    else:
        number = 0
    return number


def generate_number_distribution(rand_ob, param, limit):
    '''
    Generates triangular distribution for parameters like fuel cost
    
    Parameters
    ----------
    rand_ob : OBJECT OF RANDOM GENERATOR CLASS
        Function to calculate random values or sets
    param : VALUE
        mean value for triangular distribution
    limit : VALUE
        deviation for triangular distribution (left and right symmetric)

    Returns
    -------
    number : VALUE
        Random number generated

    '''
    #simetric triangular
    a = param * (1 - limit)
    b = param
    c = param * (1 + limit)
    number = rand_ob.dist_triangular(a,b,c)
    return number


def update_forecast(generators, forecast_df, instance_data):
    """
   This function is used to update the hourly generation and variable
   cost data for renewable energies when there is a change in the forecast 

    Parameters
    ----------
    generators : DICTIONARY
    forecast_df : DATAFRAME
    instance_data : DICTIONARY


    Returns
    -------
    generators_dict : DICTIONARY UPDATED

    """
    generators_dict = copy.deepcopy(generators)
    irr = irradiance_panel (forecast_df, instance_data)
    for gen in generators_dict.values():
        if (gen.tec == 'S'):
            gen.solar_generation( forecast_df['t_ambt'], irr, instance_data["G_stc"], 0)
            gen.solar_cost()
        if (gen.tec == 'W'):
            gen.eolic_generation(forecast_df['Wt'], instance_data["h2"],
                                     instance_data["coef_hel"], 0)
            gen.eolic_cost()

 
    return generators_dict

'SOLAR'  
    
def irradiance_panel (forecast_df, instance_data):
    '''
    This function calculates the irradiance for the solar panel 
    which is a dataframe that contains information such as global horizontal 
    irradiance (GHI), diffuse horizontal irradiance (DHI), 
    direct normal irradiance (DNI), and time; and instance_data,
    which contains information such as the tilted angle of the module,
    the module azimuth, the time zone, longitude, latitude, and other parameters.

    The function first checks if the sum of GHI and DHI is greater than 0. 
    If not, it sets the irradiance data to be equal to DNI. If it is greater than 0,
    the function calculates the total irradiance on the panel by using 
    a number of intermediate calculations such as solar altitude,
    solar azimuth, cosine of the incidence angle, sky view factor, 
    diffuse irradiance, ground irradiance, and direct irradiance. 
    It then returns the calculated irradiance in a dataframe with columns t and irr.

    Parameters
    ----------
    forecast_df : DATAFRAME
    instance_data : DICTIONARY

    Returns
    -------
    irr : DATAFRAME
        total radiation received by the solar generator, 
        as the sum of the three radiations. (DHI, DNI, GHI)

    '''
    if (forecast_df['GHI'].sum() <= 0 or forecast_df['DHI'].sum() <= 0):
        #Default only DNI if it is not GHI or DHI
        irr_data = forecast_df['DNI']
    else:       
        theta_M = instance_data["tilted_angle"]
        a_M = 90 - theta_M
        A_M = instance_data["module_azimuth"]
        TZ = instance_data["time_zone"]
        long = instance_data["longitude"]
        latit = instance_data["latitude"]
        alpha = instance_data["alpha_albedo"]
        SF1 = instance_data['shading factor']
        irr_data = {}
        for t in list(forecast_df['t'].index.values):
            LT = forecast_df['t'][t]
            DNI = forecast_df['DNI'][t] #Direct normal Irradiance
            DHI = forecast_df['DHI'][t] #Diffuse Horizontal Irradiance
            GHI = forecast_df['GHI'][t] #Global horizontal Irradiance
            day = forecast_df['day'][t] #Day of the year
            Gs = get_solar_parameters(LT, TZ, day, long, latit) #Sum altitude and sum Azimuth   
            ds = cos_incidence_angle(a_M, A_M, Gs[0], Gs[1]) #COsine incidence angle
            svf = get_sky_view_factor(theta_M) #Sky view factor
            G_dr = SF1 * DNI * ds
            if G_dr < 0:
                G_dr = 0 #negative Direct Irradiance on the PV module as zero

            G_df = svf * DHI #Diffuse irradiancia
            G_alb = alpha*(1 - svf) * GHI #Groud irradiance
            irr_data[t] = G_dr + G_df + G_alb #Total irradiance
    
    irr =  pd.DataFrame(irr_data.items(), columns = ['t','irr']) 
        
    return  irr


def min_to_hms(hm):
    '''
    This function takes in a single input hm, which represents the number of minutes,
    and converts it to hours, minutes, and seconds. 
    
    It does this by first dividing hm by 60 to get the number of hours, 
    and then using the modulus operator to get the remaining minutes. 
    It then converts this decimal value of minutes to an integer value. 
    It then again uses the decimal value of minutes to get the seconds
    by multiplying the decimal with 60 and using int to get the integer 
    value of seconds. 
    

    Parameters
    ----------
    hm : VALUE
        Minutes

    Returns
    -------
    H : VALUE
        Hours
    m : VALUE
        min
    s : VALUE
        sec

    '''

    H = int(hm / 60)
    M = ((hm / 60) - H) * 60
    m = int(M)
    S = (M - m) * 60
    s = int(S)
    return H, m, s


def decimal_hour_to_hms(hd):
    '''
    This function takes in a single input hd, which represents time in decimal format,
    and converts it to hours, minutes, and seconds. 
    It does this by first getting the integer value of hd to get the number of hours,
    then subtracting that from hd to get the decimal value of minutes. 
    It then converts this decimal value of minutes to an integer value. 
    It then again uses the decimal value of minutes to get the seconds 
    by multiplying the decimal with 60 and using int to get the integer value of seconds.

    Parameters
    ----------
    hd : VALUE
        Decimal hour

    Returns
    -------
    H : VALUE
        Hour.
    M : VALUE
        Min.
    S : VALUE
        Sec.

    '''

    H = int(hd)
    m = (hd - H) * 60
    M = int(m)
    s = m - M
    S = int(s * 60)
    return H, M, S


def get_solar_parameters(LT, TZ, dia, Long, Latit):
    '''
    This function calculates the solar parameters, specifically the solar elevation and azimuth, 
    It first converts the Local Time to minutes and calculates 
    the Time Correction Factor (TC) based on the longitude and an equation 
    that uses the day of the year and the solar position. It then calculates
    the Local Solar Time (LST) by adding the TC to the Local Time in minutes.
    It then calculates the Hour Angle (HRA) using the LST 
    and the declination angle(delta) using day of the year and latitude.

    It then calculates the solar elevation using trigonometry and the solar azimuth 
    using the declination angle and other trigonometry calculation. F
    Finally, it returns the calculated solar elevation and azimuth in degrees.

    Parameters
    ----------
    LT: local time(hour)
    TZ: time zone
    dia: counted from January 1
    Long: longitude in degrees
    Latit: latitude in degrees
    sun position calculation
    ref:https://www.pveducation.org/pvcdrom/2-properties-sunlight/suns-position

    Returns
    -------
    Elevation,Azimuth in degrees

    '''

    ka = 180 / np.pi
    LSTM = 15 * TZ#Local Standard Time Meridian(LSTM)
    EoT = lambda x:9.87 * np.sin(2 * x) - 7.53 * np.cos(x) - 1.5 * np.sin(x)#x in radians
    B = lambda d:((360 / 365) * d - 81)*(np.pi / 180)
    LT1 = LT * 60 #conversion to minutes
    TC = 4 * (Long - LSTM) + EoT(B(dia))#Time Correction Factor (TC)
    LST = LT1 + (TC / 60)#The Local Solar Time (LST)
    HRA = 15 * ((LST / 60) - 12)#Hour Angle (HRA)
    delta = 23.45 * np.sin(B(dia))#declination angle (delta)
    Elevation = (np.arcsin(np.sin(delta * np.pi / 180) * np.sin(Latit * np.pi / 180)
                           + np.cos(delta * np.pi / 180) * np.cos(Latit * np.pi / 180) * np.cos(HRA * np.pi / 180)))    
    ##calculate Azimuth 
    ## asumes teta:latitude
    k_num = (np.sin(delta * np.pi / 180)*np.cos(Latit * np.pi / 180) + np.cos(delta * np.pi / 180)
             *np.sin(Latit * np.pi / 180) * np.cos(HRA * np.pi / 180))
    k_total = k_num / np.cos(Elevation)

    if abs(k_total) > 1.0:## vancouver essay
        k_total = k_total / abs(k_total)

    Azimuth = np.arccos(k_total)
    if min_to_hms(LST)[0] >= 12:#Correction after noon
        Azimuth = 2 * np.pi - Azimuth

    return Elevation * ka, Azimuth * ka 

#a_s: Sun altitude (grados)//Elevation

def cos_incidence_angle(a_M, A_M, a_s, A_s):
    '''
    This function calculates the cosine of the incidence angle between
    the sun and a solar panel. 
    
    It then calculates the angle of rotation (Ar) between the module 
    and sun using their azimuths. It then calculates the cosine of the incidence
    angle using trigonometry, by taking the dot product of the two unit vectors 
    representing the sun's and module's direction. 

    Parameters
    ----------
    a_M : VALUE
        Module altitude (degrees)
    A_M : VALUE
        Module Azimuth (degrees)
    a_s : VALUE
        Sun altitude (degrees)
    A_s : VALUE
        Sun Azimuth (degrees)

    Returns
    -------
    ct : VALUE
        Cosine of the incidence angle.

    '''
    Ar = A_M - A_s
    c1 = np.cos(a_M * np.pi / 180) * np.cos(a_s * np.pi / 180) * np.cos(Ar * np.pi / 180)
    c2 = np.sin(a_M * np.pi / 180) * np.sin(a_s * np.pi / 180)
    ct = c1 + c2
    return ct


def get_sky_view_factor(t_M):
    '''
    Free Horizont model, calculates sky view factor     

    Parameters
    ----------
    t_M : VALUE
        tilted angle of Module (Degrees)

    Returns
    -------
    svf : VALUE
        Sky view factor

    '''

    svf = (1 + np.cos(t_M * np.pi / 180)) / 2
    return svf


