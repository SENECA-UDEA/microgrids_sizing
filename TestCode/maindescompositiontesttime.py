# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""
from src.utilities import read_data, create_objects, calculate_sizingcost, create_technologies, calculate_area, calculate_energy, interest_rate
from src.utilities import fiscal_incentive, calculate_cost_data, calculate_invertercost
import src.opt as opt
from src.classes import Random_create
import pandas as pd 
from src.operators import Sol_constructor, Search_operator
from plotly.offline import plot
import copy
pd.options.display.max_columns = None
import time
import numpy as np
Solver_data = {"MIP_GAP":0.1,"TEE_SOLVER":True,"OPT_SOLVER":"gurobi"}

rows_df_time = []

for iii in range(1, 865):
    
    print(iii)
    #PARAMETROS DE LA CORRIDA - POR DEFECTO
    #lugar
    #set the same seed for every iteration
    #seed = None
    seed = 42
    rand_ob = Random_create(seed = seed)
    place  = "Providencia"
    #iteraciones
    iteraciones_run = 30
    #area, por defecto 100%
    area_run = 1
    #%probabilidad escoger cada tecnología, por defecto 25% cada una
    d_p_run = 0.25
    s_p_run = 0.25
    w_p_run = 0.25
    b_p_run = 0.25
    #tlpsp, por defecto 1
    tlpsp_run = 1
    #nse, por defecto 5%
    nse_run = 0.05
    #%sminuscost, por defecto 100%
    sminus_cost_run = 1
    #%spluscost, por defecto 100%
    splus_cost_run = 1
    #%sube o baja fuel cost, por defecto 100% es decir el mismo valor
    fuel_cost_run = 1
    #%demanda, por defecto 100%
    demanda_run = 1
    #%forectrast viento, por defecto 100%
    forecast_w_run = 1
    #%forecast solar, por defecto 100%
    forecast_s_run = 1
    #gap, por defecto 1%
    gap_run = 0.01
    #método de añadir, grasp o random
    add_function_run = "GRASP"
    #método de remover, grasp o random
    remove_function_run = "GRASP"
    #tamaño de los json, 1 = 100%, 50% se reduciría su tamaño a la mitad
    json_baterias_run = 1
    json_diesel_run = 1
    json_solar_run = 1
    json_wind_run = 1
    htime_run = 1
    #binario incentivo fiscal, 1 sí se usa, 0 no
    delta_run = 1
    #variables auxiliares
    aumento_tiempo = "False"
    estado_json_d = "Igual"
    estado_json_s = "Igual"    
    estado_json_w = "Igual"
    estado_json_b = "Igual"
    ren_cost_run = 1
    bat_cost_run = 1
    #string para poner el nombre al escenario
    add_name1 = ""
    add_name2 = ""
    add_name3 = ""
    add_name4 = ""
    add_name5 = ""
    add_name6 = ""
    #instancias con iteraciones diferentes a 100
    if (iii >= 433):
        iteraciones_run = 50


    #instancias con diferente probabilidad de añadir una tecnología
    if((iii >= 109 and iii <=216)  or (iii >= 541 and iii <=648)):
        b_p_run = 0.5
        add_name1 = '50%bat'
    elif((iii >= 325 and iii <=432)  or (iii >= 757 and iii <=864)):
        w_p_run = 0.5
        add_name1 = '50%wind'
    elif((iii >= 217 and iii <=324)  or (iii >= 649 and iii <=756)):
        s_p_run = 0.5
        add_name1 = '50%solar'
    else:
        add_name1 = "uniform"

    #instancias tlpsp    
    if (iii >= (37 + (108 * ((iii-1)//108))) and (iii <= (72 + (108 * ((iii-1)//108))))):
        tlpsp_run = 24
        add_name2 = '24tlpsp'      
    elif (iii >= (73 + (108 * ((iii-1)//108))) and (iii <= 108 * (((iii-1)//108)+1))):
        tlpsp_run = 168
        add_name2 = '168tlpsp'
    else:
        add_name2 = '1tlpsp'

    #instancias gap
    if (iii >= (1 + (9 * ((iii-1)//9))) and (iii <= (3 + (9 * ((iii-1)//9))))):
        gap_run = 0.5
        add_name3 = '50%GAP'
    elif (iii >= (4 + (9 * ((iii-1)//9))) and (iii <= (6 + (9 * ((iii-1)//9))))):
        gap_run = 0.1
        add_name3 = '10%GAP'
    else:
        gap_run = 0.2
        add_name3 = '20%GAP'

    #instancias add o grasp
    if (iii >= (19 + (36 * ((iii-1)//36))) and (iii <= 36 * (((iii-1)//36)+1))):
        remove_function_run = "random"
        add_name4 = 'RANDOM'
    else:
        add_name4 = 'Grasp'

    #instancias cambio tamaño horizonte temporal
    if (iii%3 == 0):
        aumento_tiempo = "True"
        add_name5 = '120%htime'
        htime_run = 1.2
    elif (iii%3 == 1):
        htime_run = 0.8
        add_name5 = '80%htime'
    else:
        add_name5 = '100%htime'
   
     
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
    # read data
    demand_df_fix, forecast_df_fix, generators_total, batteries_total, instance_data, fisc_data, cost_data = read_data(demand_filepath,
                                                                                                                       forecast_filepath,
                                                                                                                        units_filepath,
                                                                                                                        instanceData_filepath,
                                                                                                                        fiscalData_filepath,
                                                                                                                        costData_filepath)
   

   
   
    generators_total, batteries_total = calculate_cost_data(generators_total, batteries_total, instance_data, cost_data)
   
    
 
   
    len_total_time = len(demand_df_fix)
   
    #crear dataframe del tamaño colocado
    if (aumento_tiempo == "False"):
        demand_df = copy.deepcopy(demand_df_fix.head(int(len_total_time * htime_run)))
        forecast_df = copy.deepcopy(forecast_df_fix.head(int(len_total_time * htime_run)))
    else:
        aux_demand = copy.deepcopy(demand_df_fix)
        aux_forecast = copy.deepcopy(forecast_df_fix)
        #si aumenta el tamaño se coloca aleatoriamente datos hasta completar el valor faltante
        mean_demand = aux_demand['demand'].mean()
        desvest_demand = aux_demand['demand'].std()
        mean_wt = aux_forecast['Wt'].mean()
        desvest_wt = aux_forecast['Wt'].std()        
        mean_dni = aux_forecast['DNI'].mean()
        desvest_dni = aux_forecast['DNI'].std()  
        mean_dhi = aux_forecast['DHI'].mean()
        desvest_dhi = aux_forecast['DHI'].std()  
        mean_ghi = aux_forecast['GHI'].mean()
        desvest_ghi = aux_forecast['GHI'].std()  
        mean_sf = aux_forecast['SF'].mean()
        desvest_sf = aux_forecast['SF'].std()
        count = len(aux_demand)
       
        #empezar a llenar los datos de forecast y demanda
        for i in range(int(len_total_time * (htime_run - 1))):
            insert_demand = rand_ob.create_randomnpnormal(mean_demand, desvest_demand, 1)
            insert_wt = rand_ob.create_randomnpnormal(mean_wt, desvest_wt, 1)
            insert_dni = rand_ob.create_randomnpnormal(mean_dni, desvest_dni, 1)
            insert_dhi = rand_ob.create_randomnpnormal(mean_dhi, desvest_dhi, 1)
            insert_ghi = rand_ob.create_randomnpnormal(mean_ghi, desvest_ghi, 1)
            insert_sf = rand_ob.create_randomnpnormal(mean_sf, desvest_sf, 1)
            numero_demand = int(insert_demand[0])
            numero_wt = int(insert_wt[0])
            numero_dni = int(insert_dni[0])
            numero_dhi = int(insert_dhi[0])
            numero_ghi = int(insert_ghi[0])
            numero_sf = int(insert_sf[0])
           
            aux_demand.loc[len(aux_demand.index)] = [count,numero_demand]
            aux_forecast.loc[len(aux_forecast.index)] = [count,numero_dni,20,numero_wt,0,numero_ghi,1,numero_sf,numero_dhi]
            count = count + 1
           
        demand_df = copy.deepcopy (aux_demand)
        forecast_df = copy.deepcopy (aux_forecast)
       
       
    #multiplicar por si hay reducción o aumento
    parameter_demand = demanda_run * instance_data['demand_covered']
    demand_df['demand'] = parameter_demand  * demand_df['demand']
    forecast_df['Wt'] = forecast_w_run * forecast_df['Wt']
    forecast_df['DNI'] = forecast_s_run * forecast_df['DNI']


    if (iii >= (1 + (18 * ((iii-1)//18))) and (iii <= (9 + (18 * ((iii-1)//18))))):
        forecast_df['GHI'] = 0
        forecast_df['DHI'] = 0
        add_name6 = 'GHI0'
    else:
        add_name6 = '3RAD'
   
    default_batteries = copy.deepcopy(batteries_total)
    default_diesel = []
    default_solar = []
    default_wind = []
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
   


       
    #Calculate interest rate
    ir = interest_rate(instance_data['i_f'],instance_data['inf'])
   
       
    years_run = instance_data['years']
       
    #Calculate CRF
    CRF = (ir * (1 + ir)**(years_run))/((1 + ir)**(years_run)-1)  
   
    #Set solver settings
    Solver_data["MIP_GAP"] = gap_run
    TEE_SOLVER = True
    OPT_SOLVER = 'gurobi'
   
    #Calculate fiscal incentives
    delta = fiscal_incentive(fisc_data['credit'],
                             fisc_data['depreciation'],
                             fisc_data['corporate_tax'],
                             ir,
                             fisc_data['T1'],
                             fisc_data['T2'])
   
    if (delta_run == 0):
        delta = 1
   
    # Create objects and generation rule
    generators_dict, batteries_dict,  = create_objects(generators,
                                                       batteries,  
                                                       forecast_df,
                                                       demand_df,
                                                       instance_data)
    #create technologies
    technologies_dict, renewables_dict = create_technologies (generators_dict,
                                                              batteries_dict)
   
   
    time_f_create_data = time.time() - time_i_create_data #final time create
   
    time_i_firstsol = time.time()
    #create the initial solution operator
    sol_constructor = Sol_constructor(generators_dict,
                                batteries_dict,
                                demand_df,
                                forecast_df)
   
    if (aumento_tiempo != "False"):
        aux_gen = copy.deepcopy(generators_dict)
        
        for j in generators_dict.values():
            count = 0
            if (j.tec != 'D'):
                mean_gen = np.array(list(aux_gen[j.id_gen].gen_rule.values())).mean()
                desvest_gen = np.array(list(aux_gen[j.id_gen].gen_rule.values())).std()
                count = len(aux_gen[j.id_gen].gen_rule)
      
                #empezar a llenar los datos de forecast y demanda
                for h in range(int(len_total_time * (htime_run - 1))):
                    insert_gen = rand_ob.create_randomnpnormal(mean_gen, desvest_gen, 1)
                    numero_gen = int(insert_gen[0])
                    aux_gen[j.id_gen].gen_rule[count] = numero_gen
                    count = count + 1
                   
        generators_dict = copy.deepcopy (aux_gen)

    
    
    
    
    #auxiliar diccionario para evitar borrar datos
    aux_instance_data = copy.deepcopy(instance_data)
    aux_instance_data['amax'] = aux_instance_data['amax'] * area_run
    aux_instance_data['tlpsp'] = tlpsp_run
    aux_instance_data['nse'] = nse_run
    aux_instance_data['sminus_cost'] = aux_instance_data['sminus_cost'] * sminus_cost_run 
    aux_instance_data['splus_cost'] =  aux_instance_data['splus_cost'] * splus_cost_run 
    aux_instance_data['fuel_cost'] = aux_instance_data['fuel_cost'] * fuel_cost_run
    aux_instance_data['years'] = years_run    
   
    #create a default solution
    sol_feasible= sol_constructor.initial_solution(aux_instance_data,
                                                    technologies_dict,
                                                    renewables_dict,
                                                    delta,
                                                    Solver_data,
                                                    rand_ob,
                                                    cost_data['NSE_COST'])
   
    #if use aux_diesel asigns a big area to avoid select it again
    if ('aux_diesel' in sol_feasible.generators_dict_sol.keys()):
        generators_dict['aux_diesel'] = sol_feasible.generators_dict_sol['aux_diesel']
        generators_dict['aux_diesel'].area = 10000000
    
    tnpccrf_calc_best = calculate_sizingcost(sol_feasible.generators_dict_sol, 
                                            sol_feasible.batteries_dict_sol, 
                                            ir = ir,
                                            years = aux_instance_data['years'],
                                            delta = delta,
                                            inverter = instance_data['inverter_cost'])
   
    time_f_firstsol = time.time() - time_i_firstsol #final time
    # set the initial solution as the best so far
    sol_best = copy.deepcopy(sol_feasible)
   
    # create the actual solution with the initial soluion
    sol_current = copy.deepcopy(sol_feasible)
   
    #check the available area
   
    #nputs for the model
    movement = "Initial Solution"
    amax =  instance_data['amax'] * area_run
    N_iterations = instance_data['N_iterations']
    #df of solutions
    rows_df = []
   
    # Create search operator
    search_operator = Search_operator(generators_dict,
                                batteries_dict,
                                demand_df,
                                forecast_df)
   
   
    dict_time_iter = {}
    dict_time_remove = {}
    dict_time_add = {}
    dict_time_make = {}
    dict_time_solve = {}
    time_i_iterations = time.time()
    iter_best = 0
    #si no es factible la solución inicial no hacer nada
    if (sol_feasible.results != None):
        for i in range(int(iteraciones_run)):
            print(iii)
            time_i_range = time.time()
            rows_df.append([i, sol_current.feasible,
                            sol_current.results.descriptive['area'],
                            sol_current.results.descriptive['LCOE'],
                            sol_best.results.descriptive['LCOE'], movement])
            if sol_current.feasible == True:    
                # save copy as the last solution feasible seen
                sol_feasible = copy.deepcopy(sol_current)
                # Remove a generator or battery from the current solution
                time_i_remove = time.time()
                if (remove_function_run == 'GRASP'):
                    sol_try, remove_report = search_operator.removeobject(sol_current, CRF, delta)
                elif (remove_function_run == 'RANDOM'):
                    sol_try, remove_report = search_operator.removerandomobject(sol_current, rand_ob)
                time_f_remove = time.time() - time_i_remove #final time
                dict_time_remove[i] = time_f_remove
                movement = "Remove"
            else:
                #  Create list of generators that could be added
                list_available_bat, list_available_gen, list_tec_gen  = search_operator.available(sol_current, amax)
                if (list_available_gen != [] or list_available_bat != []):
                    # Add a generator or battery to the current solution
                    time_i_add = time.time()
                    #aumentar la probabilidad de list si se establece, colocando más peso
                    if (b_p_run == 0.5 and list_available_bat != []):
                        list_tec_gen = list_tec_gen + ['B','B']
                    if((w_p_run == 0.5) and (list_available_gen != []) and ('W' in  list_tec_gen)):
                       list_tec_gen = list_tec_gen + ['W','W']
                    if((s_p_run == 0.5) and (list_available_gen != []) and ('S' in  list_tec_gen)):
                       list_tec_gen = list_tec_gen + ['S','S']
                    if((d_p_run == 0.5) and (list_available_gen != []) and ('D' in  list_tec_gen)):
                       list_tec_gen = list_tec_gen + ['D','D']                    
                       
                    #escoger cuál función usar
                    if (add_function_run == 'GRASP'):
                        sol_try, remove_report = search_operator.addobject(sol_current, list_available_bat, list_available_gen, list_tec_gen, remove_report,  CRF, instance_data['fuel_cost'], rand_ob, delta)
                    elif (add_function_run == 'RANDOM'):
                        sol_try = search_operator.addrandomobject(sol_current, list_available_bat, list_available_gen, list_tec_gen,rand_ob)
                    movement = "Add"
                    time_f_add = time.time() - time_i_add #final time
                    dict_time_add[i] = time_f_add
                else:
                    # return to the last feasible solution
                    sol_current = copy.deepcopy(sol_feasible)
                    continue # Skip running the model and go to the begining of the for loop
           
            #si no hay nada poner uno aleatorio para evitar errores
            if (sol_try.generators_dict_sol == {} and sol_try.batteries_dict_sol == {}):
                select_ob = rand_ob.create_rand_list(list(generators_dict.keys()))
                sol_try.generators_dict_sol[select_ob] = generators_dict[select_ob]
             
            #calculate inverter cost with installed generators
            #val = instance_data['inverter_cost']#first of the functions
            #instance_data['inverter cost'] = calculate_invertercost(sol_try.generators_dict_sol,sol_try.batteries_dict_sol,val)
            
    
                
            tnpccrf_calc = calculate_sizingcost(sol_try.generators_dict_sol,
                                                sol_try.batteries_dict_sol,
                                                ir = ir,
                                                years = years_run,
                                                delta = delta,
                                                inverter = instance_data['inverter_cost'])
            time_i_make = time.time()
            
            model2 = opt.make_model_operational(generators_dict = sol_try.generators_dict_sol,
                                               batteries_dict = sol_try.batteries_dict_sol,  
                                               demand_df=dict(zip(demand_df.t, demand_df.demand)),
                                               technologies_dict = sol_try.technologies_dict_sol,  
                                               renewables_dict = sol_try.renewables_dict_sol,
                                               fuel_cost =  instance_data['fuel_cost'] * fuel_cost_run,
                                               nse =  nse_run,
                                               TNPCCRF = tnpccrf_calc,
                                               splus_cost = instance_data['splus_cost'] * splus_cost_run,
                                               sminus_cost = instance_data['sminus_cost'] * sminus_cost_run,
                                               tlpsp = tlpsp_run,
                                               nse_cost = cost_data['NSE_COST'])
            

            time_f_make = time.time() - time_i_make
            dict_time_make[i] = time_f_make
            time_i_solve = time.time()
            results, termination = opt.solve_model(model2,
                                                   Solver_data)
            time_f_solve = time.time() - time_i_solve
            dict_time_solve[i] = time_f_solve
            
            del results
       
            if termination['Temination Condition'] == 'optimal':
                sol_try.results.descriptive['LCOE'] = model2.LCOE_value.expr()
                sol_try.results = opt.Results(model2,sol_try.generators_dict_sol, sol_try.batteries_dict_sol)
                sol_try.feasible = True
                sol_current = copy.deepcopy(sol_try)
                if sol_try.results.descriptive['LCOE'] < sol_best.results.descriptive['LCOE']:
                    sol_best = copy.deepcopy(sol_try)
                    tnpccrf_calc_best = tnpccrf_calc
                    iter_best = i
            else:
                sol_try.feasible = False
                sol_try.results.descriptive['LCOE'] = None
                sol_current = copy.deepcopy(sol_try)
       
            sol_current.results.descriptive['area'] = calculate_area(sol_current)
           
            time_f_range = time.time() - time_i_range
            dict_time_iter[i] = time_f_range    
            #print(sol_current.generators_dict_sol)
            #print(sol_current.batteries_dict_sol)
            
            del termination
            del model2
            
            
        time_f_iterations = time.time() - time_i_iterations #final time
        #df with the feasible solutions
        df_iterations = pd.DataFrame(rows_df, columns=["i", "feasible", "area", "LCOE_actual", "LCOE_Best","Movement"])
       
        time_i_results = time.time()
        #print results best solution
        print(sol_best.results.descriptive)
        print(sol_best.results.df_results)
        generation_graph = sol_best.results.generation_graph(0,len(demand_df))
        #plot(generation_graph)
        percent_df = []
        energy_df = []
        renew_df = []
        total_df = []
        brand_df = []
        try:
            percent_df, energy_df, renew_df, total_df, brand_df = calculate_energy(sol_best.batteries_dict_sol, sol_best.generators_dict_sol, sol_best.results, demand_df)
        except KeyError:
            pass
        time_f_results = time.time() - time_i_results
        time_f_total = time.time() - time_i_total #final time
       
        len_solar = 0
        len_bat = 0
        len_wind = 0
        len_diesel = 0
        list_d = []
        list_s = []
        list_w = []
        list_b = []
        for i in sol_best.generators_dict_sol.values():
            if i.tec== 'S':
                len_solar = len_solar + 1
                list_s.append(i.id_gen)
            elif i.tec == 'D':
                len_diesel = len_diesel + 1
                list_d.append(i.id_gen)
            elif i.tec == 'W':
                len_wind = len_wind + 1
                list_w.append(i.id_gen)
        for i in sol_best.batteries_dict_sol.values():
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

        area_ut = sol_best.results.descriptive['area']

        cost_vopm= 0
        for i in sol_best.generators_dict_sol.values():
            cost_vopm = cost_vopm + sol_best.results.df_results[i.id_gen+'_cost'].sum()

        lpsp_mean = sol_best.results.df_results['LPSP'].mean()
        wasted_mean = sol_best.results.df_results['Wasted Energy'].sum()
       
       
        #calcular promedios de las iteraciones
        df_time_iter = pd.DataFrame(dict_time_iter.items(), columns = ['Iteration', 'Total iteration time'])
        time_iter_average = df_time_iter['Total iteration time'].mean()
        df_time_solve = pd.DataFrame(dict_time_solve.items(), columns = ['Iteration', 'Solver time'])
        time_solve_average = df_time_solve['Solver time'].mean()
        df_time_make = pd.DataFrame(dict_time_make.items(), columns = ['Iteration', 'make model time'])
        time_make_average = df_time_make['make model time'].mean()
        df_time_remove = pd.DataFrame(dict_time_remove.items(), columns = ['Iteration', 'Remove function time'])
        time_remove_average = df_time_remove['Remove function time'].mean()
        df_time_add = pd.DataFrame(dict_time_add.items(), columns = ['Iteration', 'Add function time'])
        time_add_average = df_time_add['Add function time'].mean()
       
        lcoe_export = sol_best.results.descriptive['LCOE']
        len_gen = len(sol_best.generators_dict_sol)
        len_bat = len(sol_best.batteries_dict_sol)
       

       
        #crear fila del dataframe
        name_esc = 'esc_' + str(iii) + ' ' + str(place) + ' iter: ' + str(iteraciones_run) + ' ' + str(add_name1) + ' ' + str(add_name2) + ' ' + str(add_name3) + ' ' + str(add_name4) + ' ' + str(add_name5) + ' ' + str(add_name6)
        rows_df_time.append([iii,name_esc, place, iteraciones_run, amax, tlpsp_run, nse_run, aux_instance_data['splus_cost'],
                            aux_instance_data['sminus_cost'],aux_instance_data['fuel_cost'],len(demand_df),demanda_run,
                            forecast_df['GHI'].sum(),delta,ir,years_run,forecast_w_run,forecast_s_run,
                            gap_run, add_function_run, len(default_batteries), len(default_diesel),
                            len(default_solar),len(default_wind), b_p_run, d_p_run, s_p_run,
                            w_p_run, time_f_total, time_f_create_data,time_f_firstsol, time_f_iterations,
                            time_iter_average, time_solve_average, time_make_average, time_remove_average,
                            time_add_average, time_f_results, lcoe_export, len_gen, len_bat,
                            len_diesel,len_solar,len_wind,mean_total_b,mean_total_d,mean_total_s, mean_total_w,
                            area_ut,cost_vopm,tnpccrf_calc_best,lpsp_mean,wasted_mean, iter_best])    
        del generators_dict

#dataframe completo con todas las instancias
df_time = pd.DataFrame(rows_df_time, columns=["N", "Name", "City", "Iterations", "Area","Tlpsp",
                                              "NSE","S+_cost","S-_cost", "fuel_cost","Len_demand","Demand percent",
                                              "GHI len","delta","ir","years","Forecast_wind",
                                              "Forecast_solar","gap","add_function","json batteries", "json diesel",
                                              "json solar", "json wind",
                                              "probability add batteries","probability add diesel",
                                              "probability add solar","probability add wind","TOTAL TIME",
                                              "CREATE DATA TIME", "FIRST SOLUTION TIME","ITERATIONS TIME",
                                              "ITERATIONS MEAN TIME","ITERATIONS MEAN SOLVER TIME",
                                              "ITERATIONS MEAN MAKE MODEL TIME","ITERATIONS REMOVE FUNCTION MEAN",
                                              "ITERATIONS ADD FUNCTION MEAN", "CREATE RESULTS TIME",
                                              "LCOE","LEN GENERATORS", "LEN BATTERIES","LEN DIESEL","LEN SOLAR",
                                              "LEN WIND","MEAN GENERATION BATTERY","MEAN GENERATION DIESEL",
                                              "MEAN GENERATION SOLAR","MEAN GENERATION WIND","UTILIZED AREA",
                                              "COST VOPM","TNPC","LPSP MEAN","WASTED ENERGY MEAN","BEST ITER"])




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
#sol_best.results.df_results.to_excel("resultsprueba.xlsx")
nameins='resultbestanova' + str(iii) 
# run function
multiple_dfs(dfs, 'ExecTime', 'timeprueba.xlsx')
#multiple_dfs(dfs, 'ExecTime', 'anovafinap848to864.xlsx')

   
'''

TRM = 3910
LCOE_COP = TRM * model_results.descriptive['LCOE']
sol_best.results.df_results.to_excel("resultsesc4.xlsx")
'''