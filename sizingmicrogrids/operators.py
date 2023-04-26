# -*- coding: utf-8 -*-
from sizingmicrogrids.utilities import create_technologies
from sizingmicrogrids.utilities import calculate_sizing_cost, interest_rate
import sizingmicrogrids.opt as opt
from sizingmicrogrids.classes import Solution, Diesel
import copy
import math
import pandas as pd
from sizingmicrogrids.strategies import dispatch_my_strategy, Results_my
from sizingmicrogrids.strategies import Results, dispatch_strategy


class SolConstructor():
    """class that generates the solutions for the model
    """
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
                        rand_ob,
                        cost_data,
                        type_model,
                        Solver_data = {},
                        CRF = 0,
                        my_data = {},
                        ir = 0): 

        '''
        This function is an initial solution generator for the ILS
        
        The function creates two empty dictionaries, generators_dict_sol 
        and batteries_dict_sol, to store the generator and battery 
        solutions respectively. The function then uses the instance data
        and technologies dictionary to determine the maximum capacity 
        of the diesel generators and batteries. 
        The function then sorts the generators by capacity and shortlists
        a certain number of candidates randomly. 
        
        The function then uses the shortlisted candidates to determine 
        the maximum demand that can be covered by the generator and checks
        if the generator can fit within the available area.
        If the generator can fit, the function adds it to the solution 
        and updates the demand to be covered. 
        
        The function continues this process until all demand is covered
        or there are no more generators to consider. 
        
        The function also checks for the use of solar and wind generators,
        and batteries in case of unavailability of diesel generators. 
        
        The generated solution is returned as a dictionary
        containing the selected generators and batteries that represent
        the initial feasible solution for the model,
        if there is no feasible solution, create a fictitious diesel 
        generator with high capacity but at the same time an exorbitant cost.
        
        the solution method varies depending on the model used:
        (optimization + ILS, Dispatch Strategy + ILS or 
         Multiyear Dispatch Strategy + ILS) 

        Parameters
        ----------
        instance_data : DICTIONARY
            Instance parameteres
        technologies_dict : DICTIONARY
            technologies in the instance
        renewables_dict : DICTIONARY
            renewable technologies in the instance
        delta : NUMBER - PERCENTAGE
            tax incentive
        rand_ob : OBJECT OF CLASS RAND
            generate random numbers and lists
        cost_data : DICTIONARY
            cost parameters like energy not supplied
        type_model : STRING
            [OP, DS, MY]
            OP -> Optimization model - two stage
            DS -> Dispatch Strategy model - two stage
            MY -> Multiyear model -two stage
        Solver_data : DICTIONARY, Default = {}
            parameters for the optimizer
        CRF - NUMBER - PERCENTAGE, Default = 0
            discount rate for renewable energy
        my_data : DICTIONARY, Default = {}
            stores information to calculate multiyear function such as
            degradation or fuel cost variance
        ir : NUMBER - PERCENTAGE, Default = 0
            Interest rate
                
        Returns
        -------
        sol_initial : OBJECT OF CLASS SOLUTION
            export the initial solution

        '''
        generators_dict_sol = {}
        batteries_dict_sol = {}
        #create auxiliar dict for the iterations
        auxiliar_dict_generator = {}
        #calculate the total available area
        area_available = instance_data['amax']
        alpha_shortlist = instance_data['Alpha_shortlist']
        area = 0        
        nsh = 0
        
        #Parameters
        techno = ""
        techno2 = ""
        aux_control = ""
        
        #Calculate the maximum demand that the Diesel have to covered
        demand_to_be_covered = max(self.demand_df['demand']) 
        demand_to_be_covered = demand_to_be_covered * (1 - instance_data['nse'])
        
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
        
        
        #Initial Solution according to the model
        # If model = Optimization
        if (type_model == 'OP'):
            ir = interest_rate(instance_data['i_f'], instance_data['inf'])
    
            tnpccrf_calc = calculate_sizing_cost(generators_dict_sol, 
                                                batteries_dict_sol, 
                                                ir = ir,
                                                years = instance_data['years'],
                                                delta = delta,
                                                inverter = instance_data['inverter_cost'])
            #create the initial model
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
                                               nse_cost = cost_data["NSE_COST"])  
            #solve the model
            results, termination = opt.solve_model(model, 
                                                   Solver_data)
            
            #Check feasibility
            if termination['Temination Condition'] == 'optimal': 
                sol_results = opt.Results(model,generators_dict_sol,batteries_dict_sol,'Two-Stage')
                state = 'optimal'
            else:
                state = 'Unfeasible'
        
        else:
            #For Dispatch strategy creates solution class
            sol_results = None
            sol_try = Solution(generators_dict_sol, 
                               batteries_dict_sol, 
                               technologies_dict_sol, 
                               renewables_dict_sol,
                               sol_results) 
            #If model = Dispatch Strategy
            if (type_model == 'DS'):
                #Run the dispatch strategy process
                lcoe_cost, df_results, state, time_f, nsh = dispatch_strategy(sol_try, self.demand_df,
                                                                              instance_data, cost_data, CRF, delta, rand_ob)
            #If model = Multiyeat dispatch strategy
            elif (type_model == 'MY'):
                #Run the dispatch strategy process
                lcoe_cost, df_results, state, time_f, nsh = dispatch_my_strategy(sol_try, 
                                                                                 self.demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir)  
        
        if state != 'optimal':
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
        
            
            #Solution again according to the model.
            
            if (type_model == 'OP'):
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
                                                   nse_cost = cost_data["NSE_COST"])  
    
                #solve model with auxiliar diesel
                results, termination = opt.solve_model(model, 
                                                       Solver_data)
                
                sol_results = opt.Results(model, generators_dict_sol, batteries_dict_sol,'Two-Stage')
        
            else:
                #Default solution to run
                sol_try = Solution(generators_dict_sol, 
                                   batteries_dict_sol, 
                                   technologies_dict_sol, 
                                   renewables_dict_sol,
                                   sol_results) 
                if (type_model == 'DS'):
                    #Run the dispatch strategy process with false diesel
                    lcoe_cost, df_results, state, time_f, nsh = dispatch_strategy(sol_try, self.demand_df,
                                                                                  instance_data, cost_data, CRF, delta, rand_ob)

                elif (type_model == 'MY'):
                    #Run the dispatch strategy process with false diesel
                    lcoe_cost, df_results, state, time_f, nsh = dispatch_my_strategy(sol_try, 
                                                                                     self.demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir) 
                        
        #create solution
        sol_initial = Solution(generators_dict_sol, 
                               batteries_dict_sol, 
                               technologies_dict_sol, 
                               renewables_dict_sol,
                               sol_results) 
        sol_initial.feasible = True     
        
        #creates results for dispatch models
        if (type_model == 'DS'):
            sol_initial.results = Results(sol_initial, df_results, lcoe_cost)
        elif (type_model == 'MY'):
            sol_initial.results = Results_my(sol_initial, df_results, lcoe_cost)
        
        return sol_initial, nsh


class SearchOperator():
    """class that performs the search strategies for the ILS
    """
    def __init__(self, generators_dict, batteries_dict, demand_df, forecast_df):
        self.generators_dict = generators_dict
        self.batteries_dict = batteries_dict
        self.demand_df = demand_df
        self.forecast_df = forecast_df
        
    def remove_object(self, sol_actual, delta, rand_ob, type_model, CRF = 0): 
        '''
        This function is used to remove one generator or battery 
        from a given solution. 
        
        The function iterates through the dictionary of generators 
        and batteries and for each one, calculates their operation cost, 
        investment cost, and generation. Then it calculates the relation 
        of generation to costs, and compares it to the current minimum relation, 
        updating the minimum relation and the list of worst 
        generators/batteries accordingly.

        Finally, the function selects one of the worst generators
        or batteries randomly using the "rand_ob" object,
        removes it from the solution, updates the solution's technologies 
        and renewables dictionaries, and returns the updated solution 
        and a report of the removed generator or battery's generation.

        Parameters
        ----------
        sol_actual : OBJECT OF CLASS SOLUTION
            Current solution of the model
        delta : NUMBER - PERCENTAGE
            tax incentive
        rand_ob : OBJECT OF CLASS RAND
            generate random numbers and lists
        type_model : STRING
            [OP, DS, MY]
            OP -> Optimization model - two stage
            DS -> Dispatch Strategy model - two stage
            MY -> Multiyear model -two stage
        CRF : NUMBER - PERCENTAGE
            discount rate for renewable energy
        Returns
        -------
        solution : OBJECT OF CLASS SOLUTION
            export the generated solution
        remove_report : ARRAY - PANDAS SERIE
            array that stores the power generated by the removed object
            in the time horizon

        '''
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
                #Generation
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
            
            #If the model is Multiyear does not includes the CRF
            if (type_model == 'MY'):
                relation = sum_generation / (inv_cost  + op_cost)
            else:
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
            remove_report = pd.Series(solution.results.df_results[select_ob + '_b-'].values,
                                      index = solution.results.df_results[select_ob + '_b-'].keys()).to_dict()
            
            solution.batteries_dict_sol.pop(select_ob)
        else:
            remove_report = pd.Series(solution.results.df_results[select_ob].values,
                                      index = solution.results.df_results[select_ob].keys()).to_dict()
            
            solution.generators_dict_sol.pop(select_ob)
        
        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                            , solution.batteries_dict_sol)

        
        return solution, remove_report
    
    def add_object(self, sol_actual, available_bat, available_gen, list_tec_gen,
                      remove_report, fuel_cost, rand_ob, delta, type_model, CRF = 0):
        '''
        
        This function is used to add a new generator or battery to the 
        given energy solution. 

        The function uses the dictionary of removed objects to find 
        the maximum generation in the period of the removed object 
        and selects one of the positions with that maximum value at random.

        The function then checks if batteries or generators are available
        and selects one at random. If batteries are selected, a random battery 
        from the available options is chosen and added to the solution. 
        
        If generators are selected, the function selects a generator at random
        from the available options of the same technology random selected.
        It then checks if the selected generator has a higher generation 
        than the removed object and if it meets the capacity requirements. 
        The function calculates the levelized cost of electricity (LCOE)
        and the variable operation and maintenance cost (VOPM) for the new 
        addition and compares  it to the previous costs.
        The generator with the lowest LCOE and VOPM is chosen and added to 
        the candidates solution.
        Then randomly chooses which of the four criteria to use
        (lcoe, operating cost, capacity or that it is of the same technology) 
        and chooses the best one from the selected list
        
        The function then returns the modified solution.

        Parameters
        ----------
        sol_actual : OBJECT OF CLASS SOLUTION
            Current solution of the model
        available_bat : LIST
            list of batteries that can be selected to install in the solution
        available_gen : LIST
            list of generators that can be selected to install in the solution
        list_tec_gen : LIST
            list of technologies that are present in the available lists
        remove_report : ARRAY - PANDAS SERIE
            array that stores the power generated by the removed object and
            previously installed objects that still make the function infeasible
            in the time horizon
        fuel_cost : NUMBER
        rand_ob : OBJECT OF CLASS RAND
            generate random numbers and lists
        delta : NUMBER - PERCENTAGE
            tax incentive
        type_model : STRING
            [OP, DS, MY]
            OP -> Optimization model - two stage
            DS -> Dispatch Strategy model - two stage
            MY -> Multiyear model -two stage
        CRF : NUMBER - PERCENTAGE
            discount rate for renewable energy
        Returns
        -------
        solution : OBJECT OF CLASS SOLUTION
            export the generated solution
        remove_report : ARRAY - PANDAS SERIE
            remove-report updated, adding the demand supplied 
            by the added generator or battery

        '''
    
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
                            #If the model is Multiyear does not includes the CRF
                            if (type_model == 'MY'):
                                lcoe_inf = dic.cost_up + dic.cost_r - dic.cost_s + dic.cost_fopm 
                            else:
                                lcoe_inf = (dic.cost_up + dic.cost_r - dic.cost_s + dic.cost_fopm) * CRF
                            #calculate operative cost at the position max point
                            aux_vopm = min(reference_generation, generation_tot)
                            cost_vopm = (dic.f0 * dic.DG_max + dic.f1 * aux_vopm) * fuel_cost
                        else:
                            #Operation cost with generation rule
                            generation_total = sum(dic.gen_rule.values())
                            lcoe_op =  dic.cost_vopm * generation_total
                            #If the model is Multiyear does not includes the CRF
                            if (type_model == 'MY'):
                                lcoe_inf = dic.cost_up * delta + dic.cost_r * delta - dic.cost_s + dic.cost_fopm 
                            else:
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
                    remove_report[t] = max(0, remove_report[t] - dict_total[select_ob].DG_max)
                else:
                    remove_report[t] = max(0, remove_report[t] - dict_total[select_ob].gen_rule[t])

        solution.technologies_dict_sol, solution.renewables_dict_sol = create_technologies (solution.generators_dict_sol
                                                                                            , solution.batteries_dict_sol)
        
        return solution, remove_report
    
    
    def remove_random_object(self, sol_actual, rand_ob): 
        '''
        This function removes a random generator or battery from a given solution.
        The function starts by creating a dictionary of all generators 
        and batteries in the solution. Next, it randomly selects an object
        (either a generator or a battery) and removes it from the solution. 
        After removing the object, the function updates the solution's
        technologies and renewables dictionaries.
        It also creates a dictionary object called remove_report which is the 
        report of the generator or battery that was removed from the solution.
        It returns the updated solution and the remove_report dictionary.
        
        Parameters
        ----------
        sol_actual : OBJECT OF CLASS SOLUTION
            Current solution of the model
        rand_ob : OBJECT OF CLASS RAND
            generate random numbers and lists

        Returns
        -------
        solution : OBJECT OF CLASS SOLUTION
            export the generated solution
        remove_report : ARRAY - PANDAS SERIE
            array that stores the power generated by the removed object
            in the time horizon

        '''
        solution = copy.deepcopy(sol_actual)
        dict_actual = {**solution.generators_dict_sol, **solution.batteries_dict_sol} 
        #Random selection
        select_ob = rand_ob.create_rand_list(list(dict_actual.keys()))
        #Remove selected object
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


    def add_random_object(self, sol_actual, available_bat, 
                             available_gen, list_tec_gen, rand_ob): 
        '''
        This function adds a random generator or battery to a given solution. 
        
        The function creates a dictionary of all generators and batteries.
        Next, it randomly selects either a generator or a battery to be added 
        to the solution. If there are no available batteries,
        it will select a generator, and if there are no available generators, 
        it will select a battery. If both are available,
        it will randomly select either. The function then proceeds to select 
        a specific battery or generator based on the random selection,
        and adds it to the solution. Lastly, the function updates 
        the solution's technologies and renewables dictionaries 
        and returns the updated solution.

        Parameters
        ----------
        sol_actual : OBJECT OF CLASS SOLUTION
            Current solution of the model
        available_bat : LIST
            list of batteries that can be selected to install in the solution
        available_gen : LIST
            list of generators that can be selected to install in the solution
        list_tec_gen : LIST
            list of technologies that are present in the available lists
        rand_ob : OBJECT OF CLASS RAND
            generate random numbers and lists

        Returns
        -------
        solution : OBJECT OF CLASS SOLUTION
            export the generated solution

        '''
    
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


    def available_items(self, sol_actual, amax):
        '''
        This function reviews what items are available that are not 
        currently installed. 
        
        It starts by calculating the available area by subtracting 
        the current area used from the maximum available area. 
        
        It then creates two dictionaries, "dict_total" and "dict_actual",
        which contain all generators and batteries as well as the generators
        and batteries currently installed in the solution, respectively. 
        
        The function then uses set operations to find the items that are
        in "dict_total" but not in "dict_actual" and store them 
        in the list "non_actual". The function then iterates over "non_actual"
        and checks if the area of each item is less than or equal 
        to the available area. If it is, it checks the technology of the item,
        and adds the id to the list of batteries or generators
        and also adds the technology associated
        
        Parameters
        ----------
        sol_actual : OBJECT OF CLASS SOLUTION
            Current solution of the model
        amax : VALUE
            maximum area available.

        Returns
        -------
        available_bat : LIST
            list of batteries that can be selected to install in the solution
        available_gen : LIST
            list of generators that can be selected to install in the solution
        list_tec_gen : LIST
            list of technologies that are present in the available lists

        '''
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
        #update the data        
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

