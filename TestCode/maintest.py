# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

from src.utilities import read_data, create_objects, create_technologies, calculate_energy, interest_rate
from src.utilities import fiscal_incentive, calculate_cost_data
import src.opt as opt
import pandas as pd 
from plotly.offline import plot
pd.options.display.max_columns = None
import time
import copy
from src.classes import Random_create

rows_df_time=[]

for iii in range(1, 28):
    #set the same seed for every iteration
    #seed = None
    seed = 42
    rand_ob = Random_create(seed = seed)
    #PARAMETROS DE LA CORRIDA - POR DEFECTO
    #lugar, por defecto providencia
    place  = "Providencia"
    #% área máxima respecto a original, 1 = 100%, el área no cambia, 0.8 significa que baja el 20%
    area_run = 1
    #tlpsp, 1 = 100%, es decir no cambia
    tlpsp_run = 1
    #demanda no abastecida, por defecto 5%
    nse_run = 0.05
    #costo energía desperdiciada, 1 = 100% = el mismo valor de instance data
    splus_cost_run = 1
    #costo energía desperdiciada, 1 = 100% = el mismo valor de instance data
    sminus_cost_run = 1    
    #costo de combustible, 1 = 100% = el mismo valor de instance data
    fuel_cost_run = 1
    #demanda, 1 = 100% = el mismo valor del csv, 0.8 todo el csv baja el 80%
    demanda_run = 1
    #forecast viento, 1 = 100% = el mismo valor del csv, 0.8 todo el csv baja el 80%
    forecast_w_run = 1
    #forecast solar, 1 = 100% = el mismo valor del csv, 0.8 todo el csv baja el 80%
    forecast_s_run = 1
    #gap, por defecto 1%
    gap_run = 0.01
    #tamaño json de baterías, si se coloca por ejemplo 0.5 el json se reducirá en un 50%
    json_baterias_run = 1
    #tamaño json de diesel, si se coloca por ejemplo 0.5 el json se reducirá en un 50%
    json_diesel_run = 1
    #tamaño json solar, si se coloca por ejemplo 0.5 el json se reducirá en un 50%
    json_solar_run = 1
    #tamaño json eólico, si se coloca por ejemplo 0.5 el json se reducirá en un 50%
    json_wind_run = 1
    #longitud del horizonte temporal en %, 1 es igual, es decir 8760, 0.5 se reduciría a 4380
    htime_run = 1 
    #usar o no incentivo fiscal, binario, 1 sí, 0 no
    delta_run = 1
    #Variables auxiliares por si el horizonte temporal sube o baja y por si los json cambian de tamaño
    aumento_tiempo = "False"
    estado_json_d = "Igual"
    estado_json_s = "Igual"    
    estado_json_w = "Igual"
    estado_json_b = "Igual"
    #add_name es string que ayuda a ponerle el nombre a la instancia para que sea de fácil identificación
    add_name1 = ""
    add_name2 = ""
    add_name3 = "" 
     

    #Ejemplo
    #lpsp = 24 entre (37, 72) (145,180)…patrón de repetición = 108
    #37 = valor inicial x, 108= diferencia x2-x1, y2-y1 y 72 = valor inicial y

    if (iii >= (1 + (9 * ((iii-1)//9))) and  (iii <= (3 + (9 * ((iii-1)//9))))):
        tlpsp_run = 1
        add_name2 = '1tlpsp'       
    #ejemplo (37, 108) (181, 216), (282, 324), se repite n, 2n, 3n, 4n…. entonces se usa línea dos
    elif (iii >= (7 + (9 * ((iii-1)//9))) and (iii <= (9 + (9 * ((iii-1)//9))))):
        tlpsp_run = 168
        add_name2 = '168tlpsp'
    else:
        tlpsp_run = 24
        add_name2 = '24tlpsp' 
    

    #instancias cambio tamaño horizonte temporal

    if (iii%3 == 0):
        demanda_run = 0.03
        add_name3 = '3%demand'
    elif (iii%3 == 1):
        demanda_run = 0.005
        add_name3 = '0.5%demand'
    else:
        demanda_run = 0.01
        add_name3 = '1%demand'
 
    if (iii <= 9):
        htime_run = 720
        add_name1 = 'mes'
    elif (iii <= 18):
        htime_run = 4380
        add_name1 = 'semestre'
    else:
        htime_run = 8760
        add_name1 = 'anual'
    

    # file paths local
    demand_filepath = "../data/"+place+"/demand_"+place+".csv"
    forecast_filepath = "../data/"+place+"/forecast_"+place+".csv"
    units_filepath = "../data/"+place+"/parameters_"+place+".json"
    instanceData_filepath = "../data/"+place+"/instance_data_"+place+".json"

    #fiscal Data
    fiscalData_filepath = "../data/Cost/fiscal_incentive.json"

    #cost Data
    costData_filepath = "../data/Cost/parameters_cost.json"
 
    
    
    time_i_create_data = time.time() #initial time
    time_i_total = time.time()
    
    # read data general
    demand_df_fix, forecast_df_fix, generators_total, batteries_total, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                                       forecast_filepath,
                                                                                                                        units_filepath,
                                                                                                                        instanceData_filepath,
                                                                                                                        fiscalData_filepath,
                                                                                                                        costData_filepath)
    
    
    generators_total, batteries_total = calculate_cost_data(generators_total, batteries_total, instance_data, cost_data)
    len_total_time = len(demand_df_fix)
    
    #crear dataframe del tamaño colocado
    demand_df = copy.deepcopy(demand_df_fix.head(int(htime_run)))
    forecast_df = copy.deepcopy(forecast_df_fix.head(int(htime_run)))

        
        
    #multiplicar por si hay reducción o aumento
    parameter_demand = demanda_run 
    demand_df['demand'] = parameter_demand  * demand_df['demand'] 

    
    
    default_batteries = copy.deepcopy(batteries_total)
    default_diesel = []
    default_solar = []
    default_wind = []
    
    #definir la misma semilla para que los random siempre den lo mismo
    
    #saber la tecnología de cada generador
    for i in generators_total:
        if (i['tec'] == 'D'):
            default_diesel.append(i)
        elif (i['tec'] == 'S'):
            default_solar.append(i)
        elif (i['tec'] == 'W'):
            default_wind.append(i)
    

    
    generators = default_diesel + default_solar + default_wind
    batteries = default_batteries
    nse_run = instance_data['nse']
       
 
    
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
    ir = interest_rate(instance_data['i_f'],instance_data['inf'])
    
    #Set GAP
    MIP_GAP = 0.01
    TEE_SOLVER = True
    OPT_SOLVER = 'gurobi'
    
    #Calculate fiscal incentives
    delta = fiscal_incentive(fisc_data['credit'], 
                             fisc_data['depreciation'],
                             fisc_data['corporate_tax'],
                             ir,
                             fisc_data['T1'],
                             fisc_data['T2'])
    
    
    amax = instance_data['amax'] * area_run
    time_f_create_data = time.time() - time_i_create_data #final time create
    
    time_i_make_model = time.time()
    # Create model          
    model1 = opt.make_model(generators_dict, 
                           batteries_dict, 
                           dict(zip(demand_df.t, demand_df.demand)),
                           technologies_dict, 
                           renewables_dict, 
                           amax =  amax, 
                           fuel_cost =  instance_data['fuel_cost'] * fuel_cost_run,
                           ir = ir, 
                           nse = nse_run, 
                           years = instance_data['years'],
                           splus_cost = instance_data['splus_cost']*splus_cost_run,
                           sminus_cost = instance_data['sminus_cost']*sminus_cost_run,
                           tlpsp = tlpsp_run,
                           delta = delta)    
    
    model2 = copy.deepcopy(model1)
    generators_dict_copy = copy.deepcopy(generators_dict)
    del model1 
    print("Model generated")
    
    time_f_make_model = time.time() - time_i_make_model #final time create
    # solve model 
    
    time_i_solve = time.time()
    results, termination = opt.solve_model(model2, 
                                            optimizer = OPT_SOLVER,
                                            mipgap = MIP_GAP,
                                             tee = TEE_SOLVER)
    print("Model optimized")
    time_f_solve = time.time() - time_i_solve
    
    time_i_results = time.time()
    if termination['Temination Condition'] == 'optimal': 
       model_results = opt.Results(model2, generators_dict_copy)
       print(model_results.descriptive)
       print(model_results.df_results)
       generation_graph = model_results.generation_graph(0,len(demand_df))
       #plot(generation_graph)
       percent_df = []
       energy_df = []
       renew_df = []
       total_df = []
       brand_df = []
       try:
           percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(batteries_dict, generators_dict, model_results, demand_df)
       except KeyError:
           pass
    
    del generators_dict_copy
    del model2
    del termination
    
    time_f_results = time.time() - time_i_results
    time_f_total = time.time() - time_i_total
    
    
    len_solar = 0
    len_bat = 0
    len_wind = 0
    len_diesel = 0
    list_d = []
    list_s = []
    list_w = []
    list_b = []
    for i in generators_dict.values():
        if (model_results.descriptive['generators'][i.id_gen] == 1):
            if i.tec== 'S':
                len_solar = len_solar + 1
                list_s.append(i.id_gen)
            elif i.tec == 'D':
                len_diesel = len_diesel + 1
                list_d.append(i.id_gen)
            elif i.tec == 'W':
                len_wind = len_wind + 1
                list_w.append(i.id_gen)
    len_gen = len_solar + len_wind + len_diesel            
    for i in batteries_dict.values():
        if (model_results.descriptive['batteries'][i.id_bat] == 1):
            list_b.append(i.id_bat)
    
    mean_total_d= 0
    for i in list_d:
        mean_total_d = mean_total_d + percent_df[i+'_%'].mean()

    mean_total_s= 0
    for i in list_s:
        mean_total_s = mean_total_s + percent_df[i+'_%'].mean()
    
    mean_total_w= 0
    for i in list_w:
        mean_total_w = mean_total_w + percent_df[i+'_%'].mean()

    mean_total_b= 0
    for i in list_b:
        try:
            mean_total_b = mean_total_b + percent_df[i+'_%'].mean()            
        except KeyError:
            pass

    area_ut = model_results.descriptive['area']

    cost_vopm= 0
    for i in generators_dict.values():
        if (model_results.descriptive['generators'][i.id_gen] == 1):
            cost_vopm = cost_vopm + model_results.df_results[i.id_gen+'_cost'].sum()

    lpsp_mean = model_results.df_results['LPSP'].mean()
    wasted_mean = model_results.df_results['Wasted Energy'].sum()


    lcoe_export = model_results.descriptive['LCOE']
    
    name_esc = 'esc_' + str(iii) + ' ' + str(place) + ' iter: ' + ' ' + str(add_name1) + ' ' + str(add_name2) + ' ' + str(add_name3) 
    rows_df_time.append([iii,name_esc, place, amax, tlpsp_run, nse_run, instance_data['splus_cost']*splus_cost_run,
                        instance_data['sminus_cost']*sminus_cost_run,len(demand_df),demanda_run, forecast_w_run,forecast_s_run,
                        gap_run, len(default_batteries), len(default_diesel),
                        len(default_solar),len(default_wind),
                        time_f_total, time_f_create_data, time_f_results, time_f_solve, time_f_make_model,
                        lcoe_export, len_gen, len_bat, len_diesel,len_solar,len_wind,mean_total_b,
                        mean_total_d,mean_total_s, mean_total_w, area_ut,cost_vopm,lpsp_mean,wasted_mean])    

    #dataframe completo con todas las instancias
    df_time = pd.DataFrame(rows_df_time, columns=["N", "Name", "City","Area","Tlpsp",
                                                  "NSE", "S+_cost","S-_cost", "Len_demand","demand percent", "Forecast_wind",
                                                  "Forecast_solar","gap","json batteries", "json diesel",
                                                  "json solar", "json wind",
                                                  "TOTAL TIME","CREATE DATA TIME", "RESULTS TIME","SOLVE TIME",
                                                  "MAKE MODEL TIME","LCOE","LEN GENERATORS", "LEN BATTERIES","LEN DIESEL","LEN SOLAR",
                                                  "LEN WIND","MEAN GENERATION BATTERY","MEAN GENERATION DIESEL",
                                                  "MEAN GENERATION SOLAR","MEAN GENERATION WIND","UTILIZED AREA",
                                                  "COST VOPM","LPSP MEAN","WASTED ENERGY MEAN"])

#crear Excel
def multiple_dfs(df_list, sheets, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer,sheet_name=sheets,startrow=row , startcol=0)   
        row = row + 1
    writer.save()

# list of dataframes
dfs = [df_time]
name = 'main'
# run function
multiple_dfs(dfs, 'ExecTime', 'Total_instances' + name + '.xlsx')


'''
TRM = 3910
LCOE_COP = TRM * model_results.descriptive['LCOE']
model_results.df_results.to_excel("results.xlsx") 
'''
