# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:23:49 2022
@author: scastellanos
"""
from src.utilities import create_technologies
from src.classes import Solution, Diesel
import copy
import math
import pandas as pd
from Dispatch_Estrategy.dispatchstrategy import def_strategy, d, B_plus_D_plus_Ren, D_plus_S_and_or_W, B_plus_S_and_or_W 
from Dispatch_Estrategy.dispatchstrategy import Results 

class Sol_constructor():
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df):
        self.generators_dict = generators_dict
        self.batteries_dict = batteries_dict
        self.demand_df = demand_df
        self.forecast_df = forecast_df

    def initial_solution (self, 
                          instance_data, 
                          technologies_dict, 
                          renewables_dict,
                          delta,
                          CRF,
                          rand_ob,
                          cost_data): #initial Diesel solution

    
        generators_dict_sol = {}
        batteries_dict_sol = {}
        #create auxiliar dict for the iterations
        auxiliar_dict_generator = {}
        #calculate the total available area
        area_available = instance_data['amax']
        Alpha_shortlist = instance_data['Alpha_shortlist']
        area = 0
        
        #Calculate the maximum demand that the Diesel have to covered
        demand_to_be_covered = max(self.demand_df['demand']) 
        demand_to_be_covered = demand_to_be_covered * (1 - instance_data['nse'])
        
        #parameters
        rev = ""
        rev2 = ""
        aux_control = ""
        for g in self.generators_dict.values(): 
            if g.tec == 'D':
                auxiliar_dict_generator[g.id_gen] = g.DG_max
        rev = 'D'
        #if not diesel try with solar and one battery or wind and one battery
        if (auxiliar_dict_generator == {}):
            auxiliar_dict_bat = {}
            for b in self.batteries_dict.values(): 
                auxiliar_dict_bat[b.id_bat] = b.soc_max
            if (auxiliar_dict_bat != {}):
                sorted_batteries = sorted(auxiliar_dict_bat, key=auxiliar_dict_bat.get,reverse=True) 
                rev2 = 'B'
            for g in self.generators_dict.values(): 
                if g.tec == 'S':
                    auxiliar_dict_generator[g.id_gen] = g.Ppv_stc
            rev = 'S'
            if (auxiliar_dict_generator == {}):
                for g in self.generators_dict.values(): 
                    if g.tec == 'W':
                        auxiliar_dict_generator[g.id_gen] = g.P_y
                rev = 'W'
        #sorted generator max to min capacity
        sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=True) 
        
        available_generators = True
        while available_generators == True:
            if (len(sorted_generators) == 0 or area_available <= 0):
                available_generators = False
            else:
                #shortlist candidates
                len_candidate = math.ceil(len(sorted_generators)*Alpha_shortlist)
                position = rand_ob.create_randint(0, len_candidate-1)
                f = self.generators_dict[sorted_generators[position]]
                area_gen = f.area
                #check the technology set
                if (rev == 'D'):
                    demand_gen = f.DG_max
                elif (rev == 'S'):
                    demand_gen = f.Ppv_stc
                    if (rev2 == 'B'):
                        #put first only one battery
                        f = self.batteries_dict[sorted_batteries[0]]
                        demand_gen = 0
                        area_gen = f.area
                        rev2 = ""
                        aux_control = 'B'
                elif(rev == 'W'):
                    demand_gen = f.P_y
                    if (rev2 == 'B'):
                        f = self.batteries_dict[sorted_batteries[0]]
                        rev2 = ""
                        area_gen = f.area
                        demand_gen = 0
                        aux_control = 'B'
                #check if already supplies all demand
                if (demand_to_be_covered <= 0):
                    available_generators = False
                #add the generator to the solution
                elif (area_gen <= area_available):
                    #put the first battery if the set is renewable
                    if (aux_control == 'B'):
                        batteries_dict_sol[f.id_bat] = f
                        aux_control = ""
                    else:
                        generators_dict_sol[f.id_gen] = f
                    area += f.area
                    sorted_generators.pop(position)
                    area_available = area_available - area_gen
                    demand_to_be_covered = demand_to_be_covered - demand_gen
                #the generator cannot enter to the solution
                else:
                    sorted_generators.pop(position)

                  
        
        #create the initial solution solved
        technologies_dict_sol, renewables_dict_sol = create_technologies (generators_dict_sol, 
                                                                          batteries_dict_sol)
 

        #check strategy to be used
        strategy_def = def_strategy(generators_dict = generators_dict_sol,
                            batteries_dict = batteries_dict_sol) 
        
        #default solution to use dispatch strategy
        sol_results = None
        sol_try = Solution(generators_dict_sol, 
                           batteries_dict_sol, 
                           technologies_dict_sol, 
                           renewables_dict_sol,
                           sol_results) 
        #run dispatch strategy
        if (strategy_def == "diesel"):
            lcoe_cost, df_results, state, time_f, nsh = d(sol_try, self.demand_df, instance_data, cost_data, CRF)
        elif (strategy_def == "diesel - solar") or (strategy_def == "diesel - wind") or (strategy_def == "diesel - solar - wind"):
            lcoe_cost, df_results, state, time_f, nsh  = D_plus_S_and_or_W(sol_try, self.demand_df, instance_data, cost_data,CRF,delta)
        elif (strategy_def == "battery - solar") or (strategy_def == "battery - wind") or (strategy_def == "battery - solar - wind"):
            lcoe_cost, df_results, state, time_f, nsh  = B_plus_S_and_or_W (sol_try, self.demand_df, instance_data, cost_data, CRF, delta, rand_ob)
        elif (strategy_def == "battery - diesel - wind") or (strategy_def == "battery - diesel - solar") or (strategy_def == "battery - diesel - solar - wind"):
            lcoe_cost, df_results, state, time_f, nsh  = B_plus_D_plus_Ren(sol_try, self.demand_df, instance_data, cost_data, CRF, delta, rand_ob)
        else:
            #no feasible combination
            state = 'no feasible'
        
        if state != 'optimal': 

        #create a false Diesel auxiliar
            k = {
                        "id_gen": "aux_diesel",
                        "tec": "D",
                        "br": "DDDD",
                        "area": 0.0001,
                        "cost_up": 10000000000,
                        "cost_r": 0,
                        "cost_s": 0,
                        "cost_fopm": 0,
                        "DG_min": 0,
                        "DG_max": max(self.demand_df['demand']) ,
                        "f0": 10000000000,
                        "f1": 1000000000
                }
            obj_aux = Diesel(*k.values())
            generators_dict_sol = {}
            batteries_dict_sol = {}
            generators_dict_sol['aux_diesel'] = obj_aux
            technologies_dict_sol, renewables_dict_sol = create_technologies (generators_dict_sol, 
                                                                              batteries_dict_sol)
            
            #check solution strategy
            strategy_def = def_strategy(generators_dict = generators_dict_sol,
                    batteries_dict = batteries_dict_sol) 
            #fefault solution to run
            sol_try = Solution(generators_dict_sol, 
                   batteries_dict_sol, 
                   technologies_dict_sol, 
                   renewables_dict_sol,
                   sol_results) 
            #run dispatch strategy with false diesel
            if (strategy_def == "diesel"):
                lcoe_cost, df_results, state, time_f, nsh = d(sol_try, self.demand_df, instance_data, cost_data, CRF)
            else:
                state = 'no feasible'

        #create initial solution
        sol_initial = Solution(generators_dict_sol, 
                               batteries_dict_sol, 
                               technologies_dict_sol, 
                               renewables_dict_sol,
                               sol_results) 
        
        #if state == 'optimal': 
        sol_initial.results = Results(sol_initial, df_results, lcoe_cost)
        sol_initial.feasible = True
        
        
        
        return sol_initial


class Search_operator():
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df):
        self.generators_dict = generators_dict
        self.batteries_dict = batteries_dict
        self.demand_df = demand_df
        self.forecast_df = forecast_df
        
    def removeobject(self, sol_actual, CRF, delta): #remove one generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
        min_relation = math.inf
        #Check which one generates less energy at the highest cos}
        for dic in dict_actual.values(): 
            if dic.tec == 'B':
                #Operation cost
                op_cost = 0 
                #Investment cost
                inv_cost = d.cost_up * delta + d.cost_r * delta - d.cost_s + d.cost_fopm
                #inv_cost = (d.cost_up * delta + d.cost_r - d.cost_s)*(1+i) 
                #inv_cost2 = d.cost_fopm * ((((inf)**t_years)-1)/inf)
                sum_generation = solution.results.df_results[d.id_bat+'_b-'].sum(axis = 0, skipna = True)          
            else:
                if dic.tec == 'D':
                    sum_generation = solution.results.df_results[dic.id_gen].sum(axis = 0, skipna = True)
                    op_cost = solution.results.df_results[dic.id_gen+'_cost'].sum(axis = 0, skipna = True)
                    #op_cost *= ((((inf + txfc)**t_years)-1)/(inf+txfc))
                    inv_cost = dic.cost_up + dic.cost_r - dic.cost_s + dic.cost_fopm 
                    #inv_cost = (d.cost_up * delta + d.cost_r - d.cost_s)*(1+i) 
                    #inv_cost2 = d.cost_fopm * ((((inf)**t_years)-1)/inf)
                else:
                    sum_generation = sum(dic.gen_rule.values())
                    op_cost = dic.cost_rule
                    #op_cost *= ((((inf)**t_years)-1)/inf)
                    inv_cost = dic.cost_up * delta + dic.cost_r * delta - dic.cost_s + dic.cost_fopm 
                    #inv_cost = (d.cost_up * delta + d.cost_r - d.cost_s)*(1+i) 
                    #inv_cost2 = d.cost_fopm * ((((inf)**t_years)-1)/inf)
            relation = sum_generation / (inv_cost * CRF + op_cost)
            #relation = sum_generation * t_years / (inv_cost + inv_cost2 + op_cost)
            #Quit the worst
            if relation <= min_relation:
                min_relation = relation
                if dic.tec == 'B':
                    select_ob = dic.id_bat
                else:
                    select_ob = dic.id_gen
                
        if dict_actual[select_ob].tec == 'B':
            remove_report =  pd.Series(solution.results.df_results[select_ob+'_b-'].values,index=solution.results.df_results[select_ob+'_b-'].keys()).to_dict()
            solution.batteries_dict_sol.pop(select_ob)
        else:
            remove_report =  pd.Series(solution.results.df_results[select_ob].values,index=solution.results.df_results[select_ob].keys()).to_dict()
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)

        
        return solution, remove_report
    
    def addobject(self, sol_actual, available_bat, available_gen, list_tec_gen, remove_report, CRF, fuel_cost, rand_ob, delta): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        #get the maximum generation of removed object
        val_max = max(remove_report.values())
        #get all the position with the same maximum value
        list_max = [k for k,v in remove_report.items() if v == val_max]
        #random select: one position with the maximum value
        pos_max = rand_ob.create_rand_list(list_max)
        #get the generation in the period of maximum selected
        gen_reference = remove_report[pos_max]
        dict_total = {**self.generators_dict,**self.batteries_dict}
        best_cost = math.inf
        #generation_total: parameter to calculate the total energy by generator
        generation_total = 0
        #random select battery or generator
        if available_bat == []:
            rand_tec = rand_ob.create_rand_list(list_tec_gen)
        elif available_gen == []:
            rand_tec = 'B'
        else:
            rand_tec = rand_ob.create_rand_list(list_tec_gen + ['B'])

 
        if rand_tec == "B":
            #select a random battery
            select_ob = rand_ob.create_rand_list(available_bat)
            solution.batteries_dict_sol[select_ob] = dict_total[select_ob]
            #check generation in max period that covers the remove object
        else:
            list_rand_tec = []
            list_best_lcoe = []
            select_ob = ""
            for i in available_gen:
                dic = dict_total[i]
                #tecnhology = random
                if dic.tec == rand_tec:
                    list_rand_tec.append(dic.id_gen)
                    if dic.tec == 'D':
                        gen_generator = dic.DG_max
                    else:
                        gen_generator = dic.gen_rule[pos_max]
                    
                    #check if is better than dict remove
                    if gen_generator > gen_reference:
                        if dic.tec == 'D':
                            #Operation cost at maximum capacity
                            generation_total = len(remove_report) * dic.DG_max
                            lcoe_op =  (dic.f0 + dic.f1)*dic.DG_max*fuel_cost * len(remove_report)
                            lcoe_inf = (dic.cost_up + dic.cost_r - dic.cost_s + dic.cost_fopm) * CRF
                        else:
                            #Operation cost with generation rule
                            generation_total = sum(dic.gen_rule.values())
                            lcoe_op =  dic.cost_vopm * generation_total
                            lcoe_inf = (dic.cost_up * delta + dic.cost_r * delta - dic.cost_s + dic.cost_fopm) * CRF
                        total_lcoe = (lcoe_inf + lcoe_op)/generation_total
                        if total_lcoe <= best_cost:
                            best_cost = total_lcoe
                            list_best_lcoe.append(dic.id_gen)
                            
            if (list_best_lcoe != []):
                select_ob = rand_ob.create_rand_list(list_best_lcoe)
            else:
                select_ob = rand_ob.create_rand_list(list_rand_tec)
   
                    
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 
            #update the dictionary
            for t in remove_report.keys():
                if dict_total[select_ob].tec == 'D':
                    remove_report[t] = max(0,remove_report[t]- dict_total[select_ob].DG_max)
                else:
                    remove_report[t] = max(0,remove_report[t]- dict_total[select_ob].gen_rule[t])



        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)
        
        return solution, remove_report
    
    def addrandomobject(self, sol_actual, available_bat, available_gen, list_tec_gen, rand_ob): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_total = {**self.generators_dict,**self.batteries_dict}
        #random select battery or generator
        if available_bat == []:
            rand_tec = rand_ob.create_rand_list(list_tec_gen)

        elif available_gen == []:
            rand_tec = 'B'
        else:
            rand_tec = rand_ob.create_rand_list(list_tec_gen + ['B'])
    
 
        if rand_tec == "B":
            #select a random battery
            select_ob = rand_ob.create_rand_list(available_bat)
            solution.batteries_dict_sol[select_ob] = dict_total[select_ob]
            #check generation in max period that covers the remove object
        else:
            list_rand_tec = []
            for i in available_gen:
                dic = dict_total[i]
                #tecnhology = random
                if dic.tec == rand_tec:
                    list_rand_tec.append(dic.id_gen)

            select_ob = rand_ob.create_rand_list(list_rand_tec)
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 

        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)
       
        
        return solution
    
    def removerandomobject(self, sol_actual, rand_ob): #remove generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol} 


        select_ob = rand_ob.create_rand_list(list(dict_actual.keys()))
        #delete select object
        if dict_actual[select_ob].tec == 'B':
            remove_report =  pd.Series(solution.results.df_results[select_ob+'_b-'].values,index=solution.results.df_results[select_ob+'_b-'].keys()).to_dict()
            solution.batteries_dict_sol.pop(select_ob)
        else:
            remove_report =  pd.Series(solution.results.df_results[select_ob].values,index=solution.results.df_results[select_ob].keys()).to_dict()
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)

        
        return solution, remove_report

    def available(self, sol_actual, amax):
        solution = copy.deepcopy(sol_actual)
        available_area = amax - sol_actual.results.descriptive['area']
        list_available_gen = []
        list_available_bat = []
        list_tec_gen = []
        dict_total = {**self.generators_dict,**self.batteries_dict}
        list_keys_total = dict_total.keys()
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}     
        list_keys_actual = dict_actual.keys()
        #Check the object that is not in the current solution
        non_actual = list(set(list_keys_total) - set(list_keys_actual))
        
        #create list
        for i in non_actual: 
            g = dict_total[i]
            if g.area <= available_area:
                if g.tec == 'B':
                    list_available_bat.append(g.id_bat)
                else:
                    list_available_gen.append(g.id_gen)
                    if not (g.tec in list_tec_gen):
                        list_tec_gen.append(g.tec)
   
        return list_available_bat, list_available_gen, list_tec_gen  
# -*- coding: utf-8 -*-

