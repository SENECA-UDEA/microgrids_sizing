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

class Operators():
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df,):
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
                inv_cost = d.cost_up + d.cost_r + d.cost_om- d.cost_s
                sum_generation = solution.results.df_results[d.id_bat+'_b-'].sum(axis = 0, skipna = True)          
            else:
                sum_generation = solution.results.df_results[d.id_gen].sum(axis = 0, skipna = True)
                op_cost = solution.results.df_results[d.id_gen+'_cost'].sum(axis = 0, skipna = True)
                inv_cost = d.cost_up*d.n + d.cost_r*d.n + d.cost_om*d.n- d.cost_s*d.n
                
            relation = sum_generation / (inv_cost + op_cost)
            if relation <= min_relation:
                min_relation = relation
                if d.tec == 'B':
                    select_ob = d.id_bat
                else:
                    select_ob = d.id_gen
        
        dic_remove =  pd.Series(solution.covered_demand[select_ob].values,index=solution.covered_demand[select_ob].keys()).to_dict()

        if dict_actual[select_ob].tec == 'B':
            solution.batteries_dict_sol.pop(select_ob)
        else:
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)
        
        return solution, dic_remove
    


    def calculate_demand_covered(self, sol_actual, demand_df):
        #check the demand covered by each feasible object
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}     
        covered = {k : [0]*len(solution.results.df_results['demand']) for k in dict_actual}
        for k in dict_actual.values():
            for t in list(solution.results.df_results['demand'].index.values):
                #Check demand vs generation to avoid wasted energy in the equation
                if (k.tec == 'B'):
                    covered[k.id_bat][t] = min(solution.results.df_results[k.id_bat+'_b-'][t], solution.results.df_results['demand'][t])
                else:
                    covered[k.id_gen][t] = min(solution.results.df_results[k.id_gen][t], solution.results.df_results['demand'][t])

        covered_df = pd.DataFrame(covered, columns=[*covered.keys()])
        return covered_df
    
        
    

    def addobject(self, sol_actual, available_bat, available_gen, dic_remove, demand_df): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        pos_max = max(dic_remove, key=dic_remove.get)
        gen_max = dic_remove[pos_max]
        dict_total = {**self.generators_dict,**self.batteries_dict}
        best_option = 0
        best_cost = math.inf
        #random select battery or generator
        if available_gen == []:
            tec_select = "Battery"
        elif available_bat == []:
            tec_select = "Generator"
        else:
            #same probability by each technology
            set_select = ["Generator","Generator","Generator", "Battery"]
            tec_select = random.choice(set_select)
        
        if tec_select == "Battery":
            #select a random battery
            select_ob = random.choice(available_bat)
            solution.batteries_dict_sol[select_ob] = dict_total[select_ob]
        
        #check generation in max period that covers the remove object
        elif tec_select == "Generator":
            for i in available_gen:
                dic = dict_total[i]
                if dic.tec == 'D':
                    gen_t = dic.G_max
                else:
                    gen_t = dic.gen_rule[pos_max]
                
                dif = min(gen_max, gen_t)
                
                if dif > best_option:
                    best_option = dif
                    select_ob = dic.id_gen
                elif dif == best_option:
                    inv_cost = dic.cost_up + dic.cost_r + dic.cost_om- dic.cost_s
                    if inv_cost <= best_cost:
                        best_cost = inv_cost
                        best_option = dif
                        select_ob = dic.id_gen
                
                
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 
            #update the dictionary
            for t in list(demand_df['demand'].index.values):
                if dict_total[select_ob].tec == 'D':
                    dic_remove[t] = max(0,dic_remove[t]- dict_total[select_ob].G_max)
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

    def calculate_area (self, sol_actual):
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
        area = 0
        for i in dict_actual.values():
            area += i.area
        return area
    
    
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
    
    def initial_solution (self, 
                          instance_data,
                          generators_dict, 
                          batteries_dict, 
                          technologies_dict, 
                          renewables_dict): #initial Diesel solution
        
        generators_dict_sol = {}
        area = 0
        for g in self.generators_dict.values(): 
            if g.tec == 'D':
                generators_dict_sol[g.id_gen] = g
                area += g.area
        
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
        
        dict_actual = {**generators_dict_sol,**batteries_dict_sol}     

        if termination['Temination Condition'] == 'optimal': 
            sol_results = opt.Results(model)
            covered = {k : [0]*len(sol_results.df_results['demand']) for k in dict_actual}
            for k in dict_actual.values():
                for t in list(sol_results.df_results['demand'].index.values):
                    if (k.tec == 'B'):
                        covered[k.id_bat][t] = min(sol_results.df_results[k.id_bat][t], sol_results.df_results['demand'][t])
                    else:
                        covered[k.id_gen][t] = min(sol_results.df_results[k.id_gen][t], sol_results.df_results['demand'][t])

            covered_df = pd.DataFrame(covered, columns=[*covered.keys()])
            sol_covered_demand = covered_df
        else: 
            sol_results = None
            sol_covered_demand = None
        
        sol_initial = Solution(generators_dict_sol, 
                               batteries_dict_sol, 
                               technologies_dict_sol, 
                               renewables_dict_sol,
                               sol_results,
                               sol_covered_demand) 
        sol_initial.feasible = True

        
        return sol_initial
