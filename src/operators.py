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
                
        solution.results.descriptive['area'] = solution.results.descriptive['area'] - dict_actual[select_ob].area
        if dict_actual[select_ob].tec == 'B':
            solution.batteries_dict_sol.pop(select_ob)
        else:
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)

        
        return solution
    
    def addobject(self, sol_actual, availables, demand_df): #add generator or battery
        solution = copy.deepcopy(sol_actual)
        dict_total = {**self.generators_dict,**self.batteries_dict}
        min_relation = math.inf
        for i in availables:
            generation = 0
            dic = dict_total[i]
            if dic.tec == 'B':
                inv_cost = dic.cost_up + dic.cost_r + dic.cost_om- dic.cost_s
                #the maximum load posible - i divide into 2 because i can't charge and discharge at the same time
                generation = (len(demand_df)/2)*(dic.soc_max - dic.soc_min)
            else:
                inv_cost = dic.cost_up + dic.cost_r + dic.cost_om- dic.cost_s
                #check the power that can be supplied by the generator.
                for t in list(demand_df['demand'].index.values):
                    if dic.tec != 'D':
                        aux = min(demand_df['demand'][t], dic.gen_rule[t])
                        generation += aux
                    else:
                        aux = min(demand_df['demand'][t], dic.G_max)
                        generation += aux
            relation = generation / inv_cost
            if relation <= min_relation:
                min_relation = relation
                if dic.tec == 'B':
                    select_ob = dic.id_bat
                else:
                    select_ob = dic.id_gen          
                
        if dict_total[select_ob].tec == 'B':
            solution.batteries_dict_sol[select_ob] = dict_total[select_ob]
        else:
            solution.generators_dict_sol[select_ob] = dict_total[select_ob] 
        
        solution.results.descriptive['area'] = solution.results.descriptive['area'] + dict_total[select_ob].area  
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                              , solution.batteries_dict_sol)
        return solution
    
    
    def available(self, sol_actual, amax):
        solution = copy.deepcopy(sol_actual)
        available_area = amax - sol_actual.results.descriptive['area']
        list_available = []
        dict_total = {**self.generators_dict,**self.batteries_dict}
        list_keys_total = dict_total.keys()
        dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}     
        list_keys_actual = dict_actual.keys()
        non_actual = list(set(list_keys_total) - set(list_keys_actual))
        
        for i in non_actual: 
            g = dict_total[i]
            if g.area <= available_area:
                if g.tec == 'B':
                    list_available.append(g.id_bat)
                else:
                    list_available.append(g.id_gen)
   
        return list_available
    
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
                                           lpsp_cost = instance_data['lpsp_cost'],
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
