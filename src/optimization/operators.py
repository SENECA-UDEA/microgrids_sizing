# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:23:49 2022
@author: scastellanos
"""
from src.optimization.utilities import create_technologies
from src.optimization.utilities import calculate_sizing_cost, interest_rate
import src.optimization.opt as opt
from src.optimization.classes import Solution, Diesel
import copy
import math
import pandas as pd

class SolConstructor():
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
                          Solver_data,
                          rand_ob,
                          nse_cost): 
        
        generators_dict_sol = {}
        batteries_dict_sol = {}
        #create auxiliar dict for the iterations
        auxiliar_dict_generator = {}
        #calculate the total available area
        area_available = instance_data['amax']
        alpha_shortlist = instance_data['Alpha_shortlist']
        area = 0        
        #Calculate the maximum demand that the Diesel have to covered
        demand_to_be_covered = max(self.demand_df['demand']) 
        demand_to_be_covered = demand_to_be_covered * (1 - instance_data['nse'])
        techno = ""
        techno2 = ""
        aux_control = ""
        for g in self.generators_dict.values(): 
            if g.tec == 'D':
                auxiliar_dict_generator[g.id_gen] = g.DG_max

        techno = 'D'
        #if not diesel try with solar and one battery or wind and one battery
        if (auxiliar_dict_generator == {}):
            auxiliar_dict_bat = {}
            for b in self.batteries_dict.values(): 
                auxiliar_dict_bat[b.id_bat] = b.soc_max
            if (auxiliar_dict_bat != {}):
                sorted_batteries = sorted(auxiliar_dict_bat, 
                                          key = auxiliar_dict_bat.get, reverse = True) 
                
                techno2 = 'B'

            for g in self.generators_dict.values(): 
                if g.tec == 'S':
                    auxiliar_dict_generator[g.id_gen] = g.Ppv_stc

            techno = 'S'
            if (auxiliar_dict_generator == {}):
                for g in self.generators_dict.values(): 
                    if g.tec == 'W':
                        auxiliar_dict_generator[g.id_gen] = g.P_y

                techno = 'W'

        #sorted generator max to min capacity
        sorted_generators = sorted(auxiliar_dict_generator, 
                                   key = auxiliar_dict_generator.get, reverse = True) 
        
        available_generators = True
        while available_generators:
            if (len(sorted_generators) == 0 or area_available <= 0):
                available_generators = False
            else:
                #shortlist candidates
                len_candidate = math.ceil(len(sorted_generators) * alpha_shortlist)
                position = rand_ob.create_rand_int(0, len_candidate - 1)
                f = self.generators_dict[sorted_generators[position]]
                area_gen = f.area
                #check the technology set accord to previous calculation
                if (techno == 'D'):
                    demand_supplied = f.DG_max
                elif (techno == 'S'):
                    demand_supplied = f.Ppv_stc * f.fpv
                    if (techno2 == 'B'):
                        #put first only one battery
                        f = self.batteries_dict[sorted_batteries[0]]
                        demand_supplied = 0
                        area_gen = f.area
                        techno2 = ""
                        aux_control = 'B'

                elif(techno == 'W'):
                    demand_supplied = f.P_y
                    if (techno2 == 'B'):
                        f = self.batteries_dict[sorted_batteries[0]]
                        techno2 = ""
                        area_gen = f.area
                        demand_supplied = 0
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
                    demand_to_be_covered = demand_to_be_covered - demand_supplied
                #the generator cannot enter to the solution
                else:
                    sorted_generators.pop(position)
      
        #create the initial solution solved
        technologies_dict_sol, renewables_dict_sol = create_technologies (generators_dict_sol, 
                                                                          batteries_dict_sol)
        ir = interest_rate(instance_data['i_f'], instance_data['inf'])
        #calculate inverter cost with installed generators
        #val = instance_data['inverter_cost']#first of the functions
        #instance_data['inverter cost'] = calculate_inverter_cost(generators_dict_sol,batteries_dict_sol,val)
                
        tnpccrf_calc = calculate_sizing_cost(generators_dict_sol, 
                                            batteries_dict_sol, 
                                            ir = ir,
                                            years = instance_data['years'],
                                            delta = delta,
                                            inverter = instance_data['inverter_cost'])
        
        model = opt.make_model_operational(generators_dict = generators_dict_sol,
                                           batteries_dict = batteries_dict_sol,  
                                           demand_df = dict(zip(self.demand_df.t, self.demand_df.demand)), 
                                           technologies_dict = technologies_dict_sol,  
                                           renewables_dict = renewables_dict_sol,
                                           fuel_cost =  instance_data['fuel_cost'],
                                           nse =  instance_data['nse'], 
                                           TNPCCRF = tnpccrf_calc,
                                           splus_cost = instance_data['splus_cost'],
                                           tlpsp = instance_data['tlpsp'],
                                           nse_cost = nse_cost)  

        results, termination = opt.solve_model(model, 
                                               Solver_data)
        
        if termination['Temination Condition'] == 'optimal': 
            sol_results = opt.Results(model,generators_dict_sol,batteries_dict_sol,'Two-Stage')
        else: 
            #create a false Diesel auxiliar if not optimal
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
            
            #calculate inverter cost with installed generators
            #val = instance_data['inverter_cost']#first of the functions
            #instance_data['inverter cost'] = calculate_inverter_cost(generators_dict_sol,batteries_dict_sol,val)
                     
            tnpccrf_calc = calculate_sizing_cost(generators_dict_sol, 
                                                batteries_dict_sol, 
                                                ir = ir,
                                                years = instance_data['years'],
                                                delta = delta,
                                                inverter = instance_data['inverter_cost'])
            
            #create model with false diesel
            model = opt.make_model_operational(generators_dict = generators_dict_sol,
                                               batteries_dict = batteries_dict_sol,  
                                               demand_df = dict(zip(self.demand_df.t, self.demand_df.demand)), 
                                               technologies_dict = technologies_dict_sol,  
                                               renewables_dict = renewables_dict_sol,
                                               fuel_cost =  instance_data['fuel_cost'],
                                               nse =  instance_data['nse'], 
                                               TNPCCRF = tnpccrf_calc,
                                               splus_cost = instance_data['splus_cost'],
                                               tlpsp = instance_data['tlpsp'],
                                               nse_cost = nse_cost)  

            #solve model with auxiliar diesel
            results, termination = opt.solve_model(model, 
                                                   Solver_data)
            
            sol_results = opt.Results(model, generators_dict_sol, batteries_dict_sol,'Two-Stage')
        
        #create solution
        sol_initial = Solution(generators_dict_sol, 
                               batteries_dict_sol, 
                               technologies_dict_sol, 
                               renewables_dict_sol,
                               sol_results) 
        sol_initial.feasible = True        
        
        return sol_initial


class SearchOperator():
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df):
        self.generators_dict = generators_dict
        self.batteries_dict = batteries_dict
        self.demand_df = demand_df
        self.forecast_df = forecast_df
        
    def remove_object(self, sol_actual, CRF, delta, rand_ob): 
        '''remove one generator or battery'''
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
        min_relation = math.inf
        list_min = []
        #Check which one generates less energy at the highest cost

        for dic in dict_actual.values(): 
            if dic.tec == 'B':
                #Operation cost
                op_cost = solution.results.df_results[dic.id_bat + '_cost'].sum(axis = 0, 
                                                                                skipna = True)
                #Investment cost
                inv_cost = dic.cost_up * delta + dic.cost_r * delta - dic.cost_s + dic.cost_fopm
                #generation
                sum_generation = solution.results.df_results[dic.id_bat + '_b-'].sum(axis = 0,
                                                                                     skipna = True)          
            elif dic.tec == 'D':
                #Generation
                sum_generation = solution.results.df_results[dic.id_gen].sum(axis = 0, 
                                                                             skipna = True)
                #Operation cost
                op_cost = solution.results.df_results[dic.id_gen + '_cost'].sum(axis = 0, 
                                                                                skipna = True)
                #Investment cost
                inv_cost = dic.cost_up + dic.cost_r - dic.cost_s + dic.cost_fopm 
            else:
                #Generation
                sum_generation = sum(dic.gen_rule.values())
                #Operation cost
                op_cost = dic.cost_rule
                #Investment cost
                inv_cost = dic.cost_up * delta + dic.cost_r * delta - dic.cost_s + dic.cost_fopm 

            relation = sum_generation / (inv_cost * CRF + op_cost)

            #check if it is the worst
            if relation < min_relation:
                min_relation = relation
                #update worst list
                list_min = []
                if dic.tec == 'B':
                    list_min.append(dic.id_bat)
                else:
                    list_min.append(dic.id_gen)
            #if equal put in the worst list
            elif relation == min_relation:
                if dic.tec == 'B':
                    list_min.append(dic.id_bat)
                else:
                    list_min.append(dic.id_gen)      
                    
        #Quit random of worst list                  
        select_ob = rand_ob.create_rand_list(list_min)
        
        if dict_actual[select_ob].tec == 'B':
            remove_report =  pd.Series(solution.results.df_results[select_ob + '_b-'].values,
                                       index = solution.results.df_results[select_ob + '_b-'].keys()).to_dict()
            
            solution.batteries_dict_sol.pop(select_ob)
        else:
            remove_report =  pd.Series(solution.results.df_results[select_ob].values,
                                       index = solution.results.df_results[select_ob].keys()).to_dict()
            
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                            , solution.batteries_dict_sol)

        
        return solution, remove_report
    
    def add_object(self, sol_actual, available_bat, available_gen, list_tec_gen,
                   remove_report, CRF, fuel_cost, rand_ob, delta):
        '''add generator or battery'''
    
        solution = copy.deepcopy(sol_actual)
        #get the maximum generation of removed object
        val_max = max(remove_report.values())
        #get all the position with the same maximum value
        list_max = [k for k, v in remove_report.items() if v == val_max]
        #random select: one position with the maximum value
        pos_max = rand_ob.create_rand_list(list_max)
        #get the generation in the period of maximum selected
        reference_generation = remove_report[pos_max]
        dict_total = {**self.generators_dict, **self.batteries_dict}
        best_cost = math.inf
        best_vopm = math.inf
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
            
        else:
            #check generation in max period that covers the remove object
            list_rand_tec = []
            list_best_lcoe = []
            list_best_vopm = []
            list_best_cap = []
            select_ob = ""
            for i in available_gen:
                dic = dict_total[i]
                #tecnhology = random
                if dic.tec == rand_tec:
                    #list1 = all generators of x technology
                    list_rand_tec.append(dic.id_gen)
                    if dic.tec == 'D':
                        generation_tot = dic.DG_max
                    else:
                        generation_tot = dic.gen_rule[pos_max]
                    
                    #check if is better than dict remove
                    if generation_tot > reference_generation:
                        #list2 = all who meets the capacity
                        list_best_cap.append(dic.id_gen)
                        if dic.tec == 'D':
                            #Operation cost at maximum capacity
                            generation_total = len(remove_report) * dic.DG_max
                            lcoe_op =  (dic.f0 + dic.f1) * dic.DG_max * fuel_cost * len(remove_report)
                            lcoe_inf = (dic.cost_up + dic.cost_r - dic.cost_s + dic.cost_fopm) * CRF
                            #calculate operative cost at the position max point
                            aux_vopm = min(reference_generation, generation_tot)
                            cost_vopm = (dic.f0 * dic.DG_max + dic.f1 * aux_vopm) * fuel_cost
                        else:
                            #Operation cost with generation rule
                            generation_total = sum(dic.gen_rule.values())
                            lcoe_op =  dic.cost_vopm * generation_total
                            lcoe_inf = (dic.cost_up * delta + dic.cost_r 
                                        * delta - dic.cost_s + dic.cost_fopm) * CRF
                            #calculate operative cost at the position max point
                            cost_vopm = generation_tot * dic.cost_vopm

                        total_lcoe = (lcoe_inf + lcoe_op) / generation_total
                        #List3  = minimum average LCOE
                        if total_lcoe < best_cost:
                            #if less update the list
                            best_cost = total_lcoe
                            list_best_lcoe = []
                            list_best_lcoe.append(dic.id_gen)
                        elif total_lcoe == best_cost:
                            #if equal add to the list
                            list_best_lcoe.append(dic.id_gen)
                        
                        #List4 = miminum operative cost at point
                        if cost_vopm < best_vopm:
                            #if less update the list
                            best_vopm = cost_vopm
                            list_best_vopm = []
                            list_best_vopm.append(dic.id_gen)
                        elif cost_vopm == best_vopm:
                            #if equal add to the list
                            list_best_vopm.append(dic.id_gen) 
                       
            #random choose which list to use
            rand_parameter = rand_ob.create_rand_list(['vopm','lcoe','cap','NA'])
            
            if (list_best_cap != []):
                #if data in the list select random
                if (rand_parameter == 'lcoe'):
                    select_ob = rand_ob.create_rand_list(list_best_lcoe)
                elif (rand_parameter == 'vopm'):
                    select_ob = rand_ob.create_rand_list(list_best_vopm)
                elif (rand_parameter == 'cap'):
                    select_ob = rand_ob.create_rand_list(list_best_cap)
                else:
                    select_ob = rand_ob.create_rand_list(list_rand_tec)                
            else:
                #if neither can satisfy position max, choose random
                select_ob = rand_ob.create_rand_list(list_rand_tec)
                       
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 
            #update the dictionary
            for t in remove_report.keys():
                if dict_total[select_ob].tec == 'D':
                    remove_report[t] = max(0, remove_report[t]- dict_total[select_ob].DG_max)
                else:
                    remove_report[t] = max(0, remove_report[t]- dict_total[select_ob].gen_rule[t])

        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                            , solution.batteries_dict_sol)
        
        return solution, remove_report
    
    def add_random_object(self, sol_actual, available_bat, 
                          available_gen, list_tec_gen, rand_ob): 
        '''add random generator or battery'''
    
        solution = copy.deepcopy(sol_actual)
        dict_total = {**self.generators_dict, **self.batteries_dict}
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
    
    def remove_random_object(self, sol_actual, rand_ob): 
        '''remove random generator or battery'''
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol, **solution.batteries_dict_sol} 
        select_ob = rand_ob.create_rand_list(list(dict_actual.keys()))
        if dict_actual[select_ob].tec == 'B':
            remove_report =  pd.Series(solution.results.df_results[select_ob + '_b-'].values,
                                       index = solution.results.df_results[select_ob + '_b-'].keys()).to_dict()
            
            solution.batteries_dict_sol.pop(select_ob)
        else:
            remove_report =  pd.Series(solution.results.df_results[select_ob].values,
                                       index = solution.results.df_results[select_ob].keys()).to_dict()
            
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                            , solution.batteries_dict_sol)

        
        return solution, remove_report

    def available_items(self, sol_actual, amax):
        '''review what is available that is not installed'''
        solution = copy.deepcopy(sol_actual)
        available_area = amax - sol_actual.results.descriptive['area']
        list_available_gen = []
        list_available_bat = []
        list_tec_gen = []
        dict_total = {**self.generators_dict, **self.batteries_dict}
        list_keys_total = dict_total.keys()
        dict_actual = {**solution.generators_dict_sol, **solution.batteries_dict_sol}     
        list_keys_actual = dict_actual.keys()
        #Check the object that is not in the current solution
        non_actual = list(set(list_keys_total) - set(list_keys_actual))
        #gill the data        
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
