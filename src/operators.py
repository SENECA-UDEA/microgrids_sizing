# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:23:49 2022
@author: pmayaduque
"""
from utilities import create_technologies, calculate_sizingcost
import opt as opt
from classes import Solution
import random as random
import copy
import math
import pandas as pd

class Sol_constructor():
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df):
        self.generators_dict = generators_dict
        self.batteries_dict = batteries_dict
        self.demand_df = demand_df
        self.forecast_df = forecast_df

    def initial_solution (self, 
                          instance_data,
                          generators_dict, 
                          batteries_dict, 
                          technologies_dict, 
                          renewables_dict): #initial Diesel solution
        
        generators_dict_sol = {}
        #create auxiliar dict for the iterations
        auxiliar_dict_generator = {}
        #calculate the total available area
        area_available = instance_data['amax']
        Alpha_shortlist = instance_data['Alpha_shortlist']
        area = 0
        
        #Calculate the maximum area that the Diesel have to covered
        demand_to_be_covered = max(self.demand_df['demand'])
        
        for g in self.generators_dict.values(): 
            if g.tec == 'D':
                auxiliar_dict_generator[g.id_gen] = g.DG_max
        
        
        sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=True) 
        
        available_generators = True
        while available_generators == True:
            if (len(sorted_generators) == 0 or area_available <= 0):
                available_generators = False
            else:
                len_candidate = math.ceil(len(sorted_generators)*Alpha_shortlist)
                position = random.randint(0, len_candidate-1)
                f = self.generators_dict[sorted_generators[position]]
                area_gen = f.area
                demand_gen = f.DG_max
                if (demand_to_be_covered <= 0):
                    available_generators = False
                elif (area_gen <= area_available):
                    generators_dict_sol[f.id_gen] = f
                    area += g.area
                    sorted_generators.pop(position)
                    area_available = area_available - area_gen
                    demand_to_be_covered = demand_to_be_covered - demand_gen
                else:
                    sorted_generators.pop(position)

                   
        batteries_dict_sol = {}
        
        technologies_dict_sol, renewables_dict_sol = create_technologies (generators_dict_sol, 
                                                                          batteries_dict_sol)
        
        tnpc_calc, crf_calc = calculate_sizingcost(generators_dict_sol, 
                                                   batteries_dict_sol, 
                                                   ir = instance_data['ir'],
                                                   years = instance_data['years'])
        
        model = opt.make_model_operational(generators_dict = generators_dict_sol,
                                           batteries_dict = batteries_dict_sol,  
                                           demand_df=dict(zip(self.demand_df.t, self.demand_df.demand)), 
                                           technologies_dict = technologies_dict_sol,  
                                           renewables_dict = renewables_dict_sol,
                                           nse =  instance_data['nse'], 
                                           TNPC = tnpc_calc,
                                           CRF = crf_calc,
                                           w_cost = instance_data['w_cost'],
                                           tlpsp = instance_data['tlpsp'])  


        results, termination = opt.solve_model(model, 
                                               optimizer = 'gurobi',
                                               mipgap = 0.02,
                                               tee = True)
        
        if termination['Temination Condition'] == 'optimal': 
            sol_results = opt.Results(model)
        else: 
            sol_results = None
        
        sol_initial = Solution(generators_dict_sol, 
                               batteries_dict_sol, 
                               technologies_dict_sol, 
                               renewables_dict_sol,
                               sol_results) 
        sol_initial.feasible = True
        
        
        return sol_initial


class Search_operator():
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df):
        self.generators_dict = generators_dict
        self.batteries_dict = batteries_dict
        self.demand_df = demand_df
        self.forecast_df = forecast_df
        
    def removeobject(self, sol_actual): #remove one generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
        min_relation = math.inf
        #Check which one generates less energy at the highest cost
        for d in dict_actual.values(): 
            if d.tec == 'B':
                op_cost = 0
                inv_cost = d.cost_up + d.cost_r - d.cost_s
                sum_generation = solution.results.df_results[d.id_bat+'_b-'].sum(axis = 0, skipna = True)          
            else:
                sum_generation = solution.results.df_results[d.id_gen].sum(axis = 0, skipna = True)
                op_cost = solution.results.df_results[d.id_gen+'_cost'].sum(axis = 0, skipna = True)
                inv_cost = d.cost_up*d.n + d.cost_r*d.n - d.cost_s*d.n
                 
            relation = sum_generation / (inv_cost + op_cost)
            if relation <= min_relation:
                min_relation = relation
                if d.tec == 'B':
                    select_ob = d.id_bat
                else:
                    select_ob = d.id_gen
                
        if dict_actual[select_ob].tec == 'B':
            dic_remove =  pd.Series(solution.results.df_results[select_ob+'_b-'].values,index=solution.results.df_results[select_ob+'_b-'].keys()).to_dict()
            solution.batteries_dict_sol.pop(select_ob)
        else:
            dic_remove =  pd.Series(solution.results.df_results[select_ob].values,index=solution.results.df_results[select_ob].keys()).to_dict()
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)

        
        return solution, dic_remove
    
    def addobject(self, sol_actual, available_bat, available_gen, dic_remove, Alpha_random_gen): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        #get the position with maximum generation of removed object
        pos_max = max(dic_remove, key=dic_remove.get)
        #get the generation in the period of maximum selected
        gen_t = dic_remove[pos_max]
        dict_total = {**self.generators_dict,**self.batteries_dict}
        best_option = math.inf
        best_cost = math.inf
        #random select battery or generator
        if available_gen == []:
            tec_select = "Battery"
        elif available_bat == []:
            tec_select = "Generator"
        else:
            #same probability by each technology
            rand_tec = random.random()
            if rand_tec < Alpha_random_gen:
                tec_select = "Generator"
            else:
                tec_select = "Battery"
        
        if tec_select == "Battery":
            #select a random battery
            select_ob = random.choice(available_bat)
            solution.batteries_dict_sol[select_ob] = dict_total[select_ob]
        
        #check generation in max period that covers the remove object
        elif tec_select == "Generator":
            for i in available_gen:
                dic = dict_total[i]
                if dic.tec == 'D':
                    gen_generator = dic.DG_max
                else:
                    gen_generator = dic.gen_rule[pos_max]
                
                #choose the value closest to what is demanded
                coverage = abs(gen_t - gen_generator)
                
                if coverage < best_option:
                    best_option = coverage
                    select_ob = dic.id_gen
                    best_cost = dic.cost_up * dic.n + dic.cost_r * dic.n - dic.cost_s * dic.n
                #If two options are equal, choose the one with the lowest cost.
                elif coverage == best_option:
                    inv_cost = dic.cost_up * dic.n + dic.cost_r * dic.n - dic.cost_s * dic.n
                    if inv_cost <= best_cost:
                        best_cost = inv_cost
                        best_option = coverage
                        select_ob = dic.id_gen
                
                
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 
            #update the dictionary
            for t in dic_remove.keys():
                if dict_total[select_ob].tec == 'D':
                    dic_remove[t] = max(0,dic_remove[t]- dict_total[select_ob].DG_max)
                else:
                    dic_remove[t] = max(0,dic_remove[t]- dict_total[select_ob].gen_rule[t])



        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)
        
        return solution, dic_remove
    
    def addrandomobject(self, sol_actual, available_bat, available_gen): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_total = {**self.generators_dict,**self.batteries_dict}
        availables = available_gen + available_bat
        select_ob = random.choice(availables)
        if dict_total[select_ob].tec == 'B':
            solution.batteries_dict_sol[select_ob] = dict_total[select_ob]
        else:
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)
        return solution
    
    def removerandomobject(self, sol_actual): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol} 
        select_ob = random.choice(list(dict_actual.keys()))
        if dict_actual[select_ob].tec == 'B':
            solution.batteries_dict_sol.pop(select_ob)
        else:
            solution.generators_dict_sol.pop(select_ob)

        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)        
                                                                                    
        return solution

    def available(self, sol_actual, amax):
        solution = copy.deepcopy(sol_actual)
        available_area = amax - sol_actual.results.descriptive['area']
        list_available_gen = []
        list_available_bat = []
        dict_total = {**self.generators_dict,**self.batteries_dict}
        list_keys_total = dict_total.keys()
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}     
        list_keys_actual = dict_actual.keys()
        non_actual = list(set(list_keys_total) - set(list_keys_actual))
        
        for i in non_actual: 
            g = dict_total[i]
            if g.area <= available_area:
                if g.tec == 'B':
                    list_available_bat.append(g.id_bat)
                else:
                    list_available_gen.append(g.id_gen)
   
        return list_available_bat, list_available_gen  