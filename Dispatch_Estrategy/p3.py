# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""
from src.utilities import read_data, create_objects, create_technologies, calculate_area, calculate_energy, interest_rate
from src.utilities import fiscal_incentive, calculate_cost_data, calculate_sizingcost, calculate_invertercost
from src.classes import Random_create
import pandas as pd 
from Dispatch_Estrategy.operatorsdispatch import Sol_constructor, Search_operator
from plotly.offline import plot
from Dispatch_Estrategy.dispatchstrategy import def_strategy, dies, B_plus_D_plus_Ren, D_plus_S_and_or_W, B_plus_S_and_or_W 
from Dispatch_Estrategy.dispatchstrategy import Results
import copy
pd.options.display.max_columns = None
import time
import numpy as np


rows_df_time = []
for jjj in range(1,193):
    for ppp in range(1, 3):    
        iii = 1

        #PARAMETROS DE LA CORRIDA - POR DEFECTO
        #lugar
        #set the same seed for every iteration
        #seed = None
        seed = 42
        rand_ob = Random_create(seed = seed)
        place  = "Providencia"
        #iteraciones
        iteraciones_run = 200
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
        #método de añadir, grasp o random
        add_function_run = "GRASP"
        #método de remover, grasp o random
        remove_function_run = "RANDOM"
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
        die_cost_run = 1
        #string para poner el nombre al escenario
        iadd_name1 = ""
        iadd_name2 = ""
        iadd_name3 = ""
        iadd_name4 = ""
        jadd_name1 = ""
        jadd_name2 = ""
        jadd_name3 = ""
        jadd_name4 = ""
        jadd_name5 = ""
        jadd_name6 = ""
        jadd_name7 = ""
        #instancias con iteraciones diferentes a 100
        #time not served best solution
        best_nsh = 0
        
        if (iii <= 36):
            iteraciones_run = 200
            iadd_name1 = iteraciones_run
        else:
            iteraciones_run  = 500
            iadd_name1 = iteraciones_run

        #instancias tlpsp    
        if (iii >= 3):
            tlpsp_run = 24
            iadd_name2 = '24tlpsp'      
        else:
            iadd_name2 = '1tlpsp'


 

        #instancias cambio tamaño horizonte temporal
        if (iii >= (9 + (12 * ((iii-1)//12))) and (iii <= (12 + (12 * ((iii-1)//12))))):
            aumento_tiempo = "True"
            iadd_name3 = '120%htime'
            htime_run = 1.2
        elif (iii >= (1 + (12 * ((iii-1)//12))) and (iii <= (4 + (12 * ((iii-1)//12))))):
            htime_run = 0.8
            iadd_name3 = '80%htime'
        else:
            iadd_name3 = '100%htime'
    
        #instancias con diferente probabilidad de añadir una tecnología
        if(iii%4==2):
            b_p_run = 0.5
            iadd_name4 = '50%bat'
        elif(iii%4==0):
            w_p_run = 0.5
            iadd_name4 = '50%wind'
        elif(iii%4==3): 
            s_p_run = 0.5
            iadd_name4 = '50%solar'
        else:
            iadd_name4 = "uniform"

            
        if (jjj <= 96):
            nse_run = 0.01
            jadd_name1 = 'nse1%'
        else:
            nse_run = 0.1
            jadd_name1 = 'nse10%'   
        
        if ((jjj <= 48) or ((jjj >=97)and(jjj <= 144))):
            jadd_name2 = "delta"    
        else:   
            delta_run = 0
            jadd_name2 = "wtOUTdelta"          
        
        if (jjj >= (1 + (48 * ((jjj-1)//48))) and (jjj <= (24 + (48 * ((jjj-1)//48))))):
            splus_cost_run = 0.007
            jadd_name3 = "s+cost0.0014"
        else: 
            splus_cost_run = 0.7
            jadd_name3 = "s+cost0.14"     
            
        if (jjj >= (1 + (24 * ((jjj-1)//24))) and (jjj <= (12 + (24 * ((jjj-1)//24))))):
            sminus_cost_run = 0.2
            jadd_name4 = "s-cost20%"
        else: 
            sminus_cost_run = 1
            jadd_name4 = "s-cost100%"     

        if (jjj >= (1 + (12 * ((jjj-1)//12))) and (jjj <= (6 + (12 * ((jjj-1)//12))))):
            area_run = 0.8
            jadd_name5 = 'area80%'
        else:
            area_run = 1.2
            jadd_name5 = 'area120%'      


        if (jjj >= (1 + (6 * ((jjj-1)//6))) and (jjj <= (3 + (6 * ((jjj-1)//6))))):    
            fuel_cost_run = 0.5
            jadd_name6 = 'fcost50%'  
        else:
            fuel_cost_run = 1.5
            jadd_name6 = 'fcost150%'    
            
        if (jjj%3 == 0):
            die_cost_run = 0.8
            jadd_name7 = 'die80%cost'
        elif (jjj%3 == 1):
            bat_cost_run = 0.8
            jadd_name7 = 'bat80%cost'
        else:
            ren_cost_run = 0.8
            jadd_name7 = 'ren80%cost'
 
             
        # file paths local
        demand_filepath = "../data/"+place+"/demand_"+place+".csv"
        forecast_filepath = "../data/"+place+"/forecast_"+place+".csv"
        instanceData_filepath = "../data/"+place+"/instance_data_"+place+".json"
        
        if (ppp <= 1):
            units_filepath = "../data/"+place+"/parameters_"+place+"_big"+".json"
        else:
           units_filepath = "../data/"+place+"/parameters_"+place+"_small"+".json"
        #fiscal Data
        fiscalData_filepath = "../data/Cost/fiscal_incentive.json"
       
        #cost Data
        costData_filepath = "../data/Cost/parameters_cost.json"
               
           
       
       
        time_i_create_data = time.time() #initial time
        time_i_total = time.time()
        # read data
        demand_df_fix, forecast_df_fix, generators_total, batteries_total, instance_data_def, fisc_data, cost_data_def = read_data(demand_filepath,
                                                                                                                           forecast_filepath,
                                                                                                                            units_filepath,
                                                                                                                            instanceData_filepath,
                                                                                                                            fiscalData_filepath,
                                                                                                                            costData_filepath)
        
       
        instance_data = copy.deepcopy(instance_data_def)
        cost_data = copy.deepcopy(cost_data_def)
        instance_data["splus_cost"] = instance_data["splus_cost"] * splus_cost_run 
        instance_data["amax"] = instance_data["amax"] * area_run 
        instance_data['tlpsp'] = tlpsp_run
        instance_data['nse'] = nse_run
        instance_data['fuel_cost'] = instance_data['fuel_cost'] * fuel_cost_run
        cost_data['NSE_COST']['L1'][1] = cost_data['NSE_COST']['L1'][1] * sminus_cost_run
        cost_data['NSE_COST']['L2'][1] = cost_data['NSE_COST']['L2'][1] * sminus_cost_run
        cost_data['NSE_COST']['L3'][1] = cost_data['NSE_COST']['L3'][1] * sminus_cost_run
        cost_data['NSE_COST']['L4'][1] = cost_data['NSE_COST']['L4'][1] * sminus_cost_run
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
                numero_demand = max(numero_demand,0)
                numero_wt = int(insert_wt[0])
                numero_wt = max(numero_wt, 0)
                numero_dni = int(insert_dni[0])
                numero_dni = max(numero_dni, 0)
                numero_dhi = int(insert_dhi[0])
                numero_dhi = max(numero_dhi, 0)
                numero_ghi = int(insert_ghi[0])
                numero_ghi = max(numero_ghi, 0)
                numero_sf = int(insert_sf[0])
                numero_sf = max(numero_sf, 0)
               
                aux_demand.loc[len(aux_demand.index)] = [count,numero_demand]
                aux_forecast.loc[len(aux_forecast.index)] = [count,numero_dni,20,numero_wt,0,numero_ghi,1,numero_sf,numero_dhi]
                count = count + 1
               
            demand_df = copy.deepcopy (aux_demand)
            forecast_df = copy.deepcopy (aux_forecast)
           
           
        #multiplicar por si hay reducción o aumento
        demand_df['demand'] = instance_data['demand_covered']  * demand_df['demand']

    
       
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
    
        if (ren_cost_run == 0.8):
            for i in default_solar:
                i['cost_up'] = i['cost_up'] * ren_cost_run
                i['cost_s'] = i['cost_s'] * ren_cost_run
                i['cost_fopm'] = i['cost_fopm'] * ren_cost_run 
            for i in default_wind:
                i['cost_up'] = i['cost_up'] * ren_cost_run
                i['cost_s'] = i['cost_s'] * ren_cost_run  
                i['cost_fopm'] = i['cost_fopm'] * ren_cost_run 
            
        if (bat_cost_run == 0.8):
            for i in default_batteries:
                i['cost_up'] = i['cost_up'] * bat_cost_run
                i['cost_s'] = i['cost_s'] * bat_cost_run  
                i['cost_r'] = i['cost_r'] * bat_cost_run 
                i['cost_fopm'] = i['cost_fopm'] * bat_cost_run  
           
        if (die_cost_run == 0.8):
            for i in default_diesel:
                i['cost_up'] = i['cost_up'] * die_cost_run
                i['cost_s'] = i['cost_s'] * die_cost_run 
                i['cost_r'] = i['cost_r'] * die_cost_run 
                i['cost_fopm'] = i['cost_fopm'] * die_cost_run   
    
    
        generators = default_diesel + default_solar + default_wind
        batteries = default_batteries
        nse_run = instance_data['nse']
       
    
    
           
        #Calculate interest rate
        ir = interest_rate(instance_data['i_f'],instance_data['inf'])
       
           
        years_run = instance_data['years']
           
        #Calculate CRF
        CRF = (ir * (1 + ir)**(years_run))/((1 + ir)**(years_run)-1)  
       
    
       
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
       

        #auxiliar diccionario para evitar borrar datos
        aux_instance_data = copy.deepcopy(instance_data)
   
       
        #create a default solution
        sol_feasible= sol_constructor.initial_solution(aux_instance_data,
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
        

       
        time_f_firstsol = time.time() - time_i_firstsol #final time
        sol_feasible.results.descriptive['area'] = calculate_area(sol_feasible)    
        # set the initial solution as the best so far
        sol_best = copy.deepcopy(sol_feasible)
        
        # create the actual solution with the initial soluion
        sol_current = copy.deepcopy(sol_feasible)

        #check the available area
       
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
                
                #calculate inverter cost with installed generators
                #instance_data['inverter cost'] = calculate_invertercost(sol_try.generators_dict_sol,sol_try.batteries_dict_sol,instance_data['inverter_cost'])
                        
                   
                tnpccrf_calc = calculate_sizingcost(sol_try.generators_dict_sol,
                                                    sol_try.batteries_dict_sol,
                                                    ir = ir,
                                                    years = years_run,
                                                    delta = delta,
                                                    inverter = instance_data['inverter_cost'])
                
                time_i_make = time.time()
                
                strategy_def = def_strategy(generators_dict = sol_try.generators_dict_sol,
                                batteries_dict = sol_try.batteries_dict_sol) 
                print("defined strategy")
                if (strategy_def == "diesel"):
                    lcoe_cost, df_results, state, time_f, nsh  = dies(sol_try, demand_df, instance_data, cost_data, CRF)
                elif (strategy_def == "diesel - solar") or (strategy_def == "diesel - wind") or (strategy_def == "diesel - solar - wind"):
                    lcoe_cost, df_results, state, time_f, nsh   = D_plus_S_and_or_W(sol_try, demand_df, instance_data, cost_data,CRF, delta )
                elif (strategy_def == "battery - solar") or (strategy_def == "battery - wind") or (strategy_def == "battery - solar - wind"):
                    lcoe_cost, df_results, state, time_f, nsh   = B_plus_S_and_or_W (sol_try, demand_df, instance_data, cost_data, CRF, delta, rand_ob)
                elif (strategy_def == "battery - diesel - wind") or (strategy_def == "battery - diesel - solar") or (strategy_def == "battery - diesel - solar - wind"):
                    lcoe_cost, df_results, state, time_f, nsh   = B_plus_D_plus_Ren(sol_try, demand_df, instance_data, cost_data, CRF, delta, rand_ob)
                else:
                    state = 'no feasible'
                    df_results = []
                
                print("finished simulation - state: " + state)
    
                time_f_make = time.time() - time_i_make
                dict_time_make[i] = time_f_make
                time_i_solve = time.time()
                time_f_solve = time.time() - time_i_solve
                dict_time_solve[i] = time_f_solve
    
                if state == 'optimal':
                    sol_try.results = Results(sol_try, df_results, lcoe_cost)
                    sol_try.feasible = True
                    sol_current = copy.deepcopy(sol_try)
                    
                    
                    #Search the best solution
                    if sol_try.results.descriptive['LCOE'] <= sol_best.results.descriptive['LCOE']:
                        sol_try.results.descriptive['area'] = calculate_area(sol_try)
                        sol_best = copy.deepcopy(sol_try)   
                        tnpccrf_calc_best = tnpccrf_calc
                        iter_best = i
                        best_nsh = nsh
                else:
                    sol_try.feasible = False
                    df_results = []
                    lcoe_cost = None
                    sol_try.results = Results(sol_try, df_results, lcoe_cost)
                    sol_current = copy.deepcopy(sol_try)
                
            
                sol_current.results.descriptive['area'] = calculate_area(sol_current)
               
                time_f_range = time.time() - time_i_range
                dict_time_iter[i] = time_f_range    
                #print(sol_current.generators_dict_sol)
                #print(sol_current.batteries_dict_sol)
                
                del df_results
                del sol_try
                
                
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
            
            if ('aux_diesel' in sol_best.generators_dict_sol.keys()):
                 len_diesel = 0
                 len_solar = 0
                 len_bat = 0
                 len_wind = 0
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
            wasted_mean = sol_best.results.df_results['Wasted Energy'].mean()
           
           
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
            name_esci = str(iii) + ' ' + str(place) + ' iter: ' + str(iteraciones_run) + ' ' + str(iadd_name1) + ' ' + str(iadd_name2) + ' ' + str(iadd_name3)+ ' ' + str(iadd_name4)  
            name_escj =  str(jjj) + ' ' + str(jadd_name1) + ' ' + str(jadd_name2)+ ' ' + str(jadd_name3)+ ' ' + str(jadd_name4)+ ' ' + str(jadd_name5)+ ' ' + str(jadd_name6)+ ' ' + str(jadd_name7)
            name_esc = 'esc_'   + name_esci + ' MODEL- ' + name_escj
            rows_df_time.append([iii,jjj,ppp,name_esc, place, iteraciones_run, amax, tlpsp_run, nse_run, aux_instance_data['splus_cost'],
                                sminus_cost_run,aux_instance_data['fuel_cost'],len(demand_df),demanda_run,
                                forecast_df['GHI'].sum(),delta,ir,years_run,forecast_w_run,forecast_s_run,
                                add_function_run, remove_function_run, len(default_batteries), len(default_diesel),
                                len(default_solar),len(default_wind), b_p_run, d_p_run, s_p_run,
                                w_p_run, time_f_total, time_f_create_data,time_f_firstsol, time_f_iterations,
                                time_iter_average, time_solve_average, time_make_average, time_remove_average,
                                time_add_average, time_f_results, lcoe_export, len_gen, len_bat,
                                len_diesel,len_solar,len_wind,mean_total_b,mean_total_d,mean_total_s, mean_total_w,
                                area_ut,cost_vopm,tnpccrf_calc_best,lpsp_mean,wasted_mean, iter_best, best_nsh])    
            del generators_dict
            del batteries_dict
            del generators_total
            del batteries_total
            del instance_data
    
            #dataframe completo con todas las instancias
            df_time = pd.DataFrame(rows_df_time, columns=["N problem","N algortihm","ppp", "Name", "City", "Iterations", "Area","Tlpsp",
                                                          "NSE","S+_cost","S-_cost", "fuel_cost","Len_demand","Demand percent",
                                                          "GHI len","delta","ir","years","Forecast_wind",
                                                          "Forecast_solar","add_function","remove-function","json batteries", "json diesel",
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
                                                          "COST VOPM","TNPC","LPSP MEAN","WASTED ENERGY MEAN","BEST ITER", "Not served hours"])
            



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
            # run function
            multiple_dfs(dfs, 'ExecTime', 'p3sizegenerators.xlsx')
            #multiple_dfs(dfs, 'ExecTime', 'anovafinap848to864.xlsx')

   
'''

TRM = 3910
LCOE_COP = TRM * model_results.descriptive['LCOE']
sol_best.results.df_results.to_excel("resultsesc4.xlsx")
'''# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

