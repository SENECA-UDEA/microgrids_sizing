# -*- coding: utf-8 -*-
'''
At the beginning are the dispatch strategies for the one-year model,
then, at the end they are repeated for the multi-year model
'''
import numpy as np
import time
import pandas as pd
import math
import copy
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'


def select_strategy (batteries_dict, generators_dict):

    '''
    This function takes in two dictionaries, one for batteries and one for generators,
    as inputs. The function initializes four variables, "d", "s", "b", and "w", 
    to keep track of the presence of the technologies (diesel, solar, 
                                                       battery, and wind) 
    It then iterates through the generators dictionary and checks the 
    technology attribute of each generator, updating the corresponding variable
    if the technology is present. 

    Afterwards, the function creates a string called "dispatch" 
    to store the strategy name based on the presence of the technologies.
    
    Finally, the function returns the "dispatch" string, 
    which is the strategy name based on the technologies present.
    
    Parameters
    ----------
    batteries_dict : DICTIONARY
    generators_dict : DICTIONARY

    Returns
    -------
    dispatch : STRING

    '''    
    #diesel
    d = 0
    #solar
    s = 0
    #battery
    b = 0
    #wind
    w = 0    
    dispatch = ""
    #check technologies to create strategy name
    for gen in generators_dict.values(): 
        if (gen.tec == 'D'):
            d = 1
        elif (gen.tec == 'S'):
            s = 1
        elif (gen.tec == 'W'):
            w = 1
    
    if (batteries_dict != {}):
        b = 1
        
    cont = 0
    #create name strategy
    if (b == 1):
        dispatch = "battery"
        cont = 1
    
    if (d == 1 and cont == 0):
        dispatch += "diesel"
    elif (d == 1):
        dispatch += " - diesel"
    
    if (s == 1):
        dispatch += " - solar"
    
    if (w == 1):
        dispatch += " - wind"
    
    return dispatch


#diesel dispatch strategy
def ds_diesel (solution, demand_df, instance_data, cost_data, CRF):
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are only Diesel generators, 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    CRF : NUMBER
        percentage to annualize costs

    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''
    #initial parameters 
    time_i = time.time()
    aux_dict = {}
    lcoe_op = 0 #operative cost
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k : [0] * LEN_DATA for k in solution.generators_dict_sol} 
    #variable cost
    cost = {k + '_cost': [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #batteries, 0 in this case, report the same of others strategies
    #battery stated of charge
    soc = {l + '_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l + '_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l + '_b-': [0]*LEN_DATA for l in solution.batteries_dict_sol} 
    fuel_cost = instance_data['fuel_cost']
    inverter = instance_data['inverter_cost']
    AVERAGE_DEMAND = np.mean(demand_df['demand'])
    nsh = 0 #count not server hours
    
    #calculate investments cost
    for g in solution.generators_dict_sol.values():  
        lcoe_inf = (g.cost_up + g.cost_r - g.cost_s + g.cost_fopm)* CRF 
        lcoe_inftot += lcoe_inf
        #assume it produces around the average
        prod = min (AVERAGE_DEMAND, g.DG_max)
        lcoe_op = (g.f0 * g.DG_max + g.f1 * prod) * fuel_cost * LEN_DATA
        aux_dict[g.id_gen] = ((prod * LEN_DATA) / (lcoe_inf * CRF + lcoe_op))

    #sort to initialize always with the best lcoe diesel generator
    sorted_generators = sorted(aux_dict, key = aux_dict.get, reverse = False) 
    
    #simulation
    for t in demand_df['t']:
        #initilialy demand to be covered is the same that demand
        demand_tobe_covered = demand_df['demand'][t]
        #supply with each generator
        for i in sorted_generators:
             gen = solution.generators_dict_sol[i]
             #if all demand covered not generate
             if (demand_tobe_covered == 0):
                 p[i][t] = 0
                 cost[i + '_cost'][t] = 0
             #if lowest that reference can generate, calculate splus                
             elif (demand_tobe_covered < gen.DG_min):
                 p[i][t] = gen.DG_min
                 ptot += p[i][t]
                 cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                 costvopm += cost[i + '_cost'][t]
                 splus['s+'][t] = (gen.DG_min - demand_tobe_covered)
                 demand_tobe_covered = 0
                 costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                 splustot += costsplus['cost_s+'][t]
             #greatest rate, supply all the demand
             elif (gen.DG_max >= demand_tobe_covered):
                 p[i][t] = demand_tobe_covered
                 ptot += p[i][t]
                 cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                 costvopm += cost[i + '_cost'][t]
                 demand_tobe_covered = 0
             else:
                #supply until rated capacity
                p[i][t] = gen.DG_max
                ptot += p[i][t]
                cost[i + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                costvopm += cost[i + '_cost'][t]
                demand_tobe_covered = demand_tobe_covered - gen.DG_max

        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t]           
    
    #mean average lpsp
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], 
                                 min_periods = None, center = False, 
                                 win_type = None, on = None, axis = 0).mean()
    
    count_lpsp = lpsp_check[lpsp_check['lpsp'] > instance_data['nse']].count()
    
    #check feasible lpsp
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'

    #create results
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)/
                 (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), 
                             columns = ['S-','LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, 
                            bplus_df, sminus_df, splus_df, generation_cost], axis = 1) 
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f, nsh
    
    

def ds_diesel_renewable (solution, demand_df, instance_data, 
                         cost_data, CRF, delta):    
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are Diesel generators and 
    renewable generators (solar panel PV - or Eolic turbine or both), 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    CRF : NUMBER
        percentage to annualize costs
    delta: NUMBER
        percentage - fiscal incentive
    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''

    #initial parameters 
    time_i = time.time()
    aux_dict = {}
    lcoe_op = 0 #operative cost
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol} 
    #variable cost
    cost = {k + '_cost': [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #batteries, 0 in this case, report the same of others strategies
    #battery stated of charge
    soc = {l + '_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l + '_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l + '_b-': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    fuel_cost = instance_data['fuel_cost']
    inverter = instance_data['inverter_cost']
    list_ren = [] #renewable generation
    demand_tobe_covered = [] 
    #initial min reference (diesel reference for renewable generators)
    min_ref = math.inf 
    AVERAGE_DEMAND = np.mean(demand_df['demand'])
    nsh = 0 #count not server hours
 
    #calculate investment cost    
    for g in solution.generators_dict_sol.values():
        if (g.tec == 'D'):
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s + g.cost_fopm) * CRF 
            lcoe_inftot += lcoe_inf   
            #assume it produces around the average
            prod = min (AVERAGE_DEMAND, g.DG_max)
            lcoe_op = (g.f0 * g.DG_max + g.f1 * prod) * fuel_cost * LEN_DATA
            aux_dict[g.id_gen] = (prod * LEN_DATA) / (lcoe_inf * CRF + lcoe_op)
            
            #get the lowest reference
            if g.DG_min <= min_ref:
                 min_ref = g.DG_min

        else:
            lcoe_inf = (g.cost_up  * delta  + g.cost_r 
                        * delta - g.cost_s + g.cost_fopm)* CRF 
            lcoe_inftot += lcoe_inf    
            list_ren.append(g.id_gen)

    #sorted diesel initial the best lcoe generator
    sorted_generators = sorted(aux_dict, key = aux_dict.get,reverse = False) 
    
    #reference is the generator diesel in the first position, 
    #it is necessary for the renewable objets
    ref = solution.generators_dict_sol[sorted_generators[0]].DG_min
    
    #simulation
    for t in demand_df['t']:
        #initialy demand to be covered is the same that demand
        demand_tobe_covered = demand_df['demand'][t]
        #check is higher than min ref
        if (demand_tobe_covered >= min_ref):
            #calculate renewable generation
            for ren in list_ren:
                renew = solution.generators_dict_sol[ren]
                p[ren][t] = renew.gen_rule[t]
                #calculate cost
                cost[ren + '_cost'][t] = p[ren][t] * renew.cost_vopm
                costvopm += cost[ren + '_cost'][t]

            #total renewable generation
            
            generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t]
                                 for i in list_ren)
            ptot += generation_ren
            #surplus energy
            if generation_ren > (demand_tobe_covered - ref):
                splus['s+'][t] = (generation_ren + ref - demand_tobe_covered)
                costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                splustot += costsplus['cost_s+'][t]
                demand_tobe_covered = ref
            #still demand to be covered    
            else:
                demand_tobe_covered = demand_tobe_covered - generation_ren
            
            #turn on only one generator at minimun load
            if (demand_tobe_covered <= ref):
                n = sorted_generators[0]
                gen = solution.generators_dict_sol[n]
                p[n][t] = ref
                ptot += p[n][t]
                cost[n+'_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[n][t]) * fuel_cost
                costvopm += cost[n + '_cost'][t]
                demand_tobe_covered = demand_tobe_covered - ref                
            else:
                #supply the demand with diesel
                for i in sorted_generators:
                     gen = solution.generators_dict_sol[i]
                     #lowest that reference, don't turn on
                     if (demand_tobe_covered < gen.DG_min):
                         p[i][t] = 0
                         cost[i + '_cost'][t] = 0
                     #rated capacity is higher than demand, then  supply all demand
                     elif (gen.DG_max >= demand_tobe_covered):
                         p[i][t] = demand_tobe_covered
                         ptot += p[i][t]
                         cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                         costvopm += cost[i + '_cost'][t]
                         demand_tobe_covered = 0
                    #suply until rated capacity
                     else:
                        p[i][t] = gen.DG_max
                        ptot += p[i][t]
                        cost[i + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                        costvopm += cost[i + '_cost'][t]
                        #update demand to be covered
                        demand_tobe_covered = demand_tobe_covered - gen.DG_max

        else:
        #use one diesel
            i = sorted_generators[0]
            gen = solution.generators_dict_sol[i]
            p[i][t] = gen.DG_min
            ptot += p[i][t]
            cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
            costvopm += cost[i + '_cost'][t]
            splus['s+'][t] = (gen.DG_min - demand_tobe_covered)
            demand_tobe_covered = 0
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            splustot += costsplus['cost_s+'][t]
                                  
        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 

    #mean average lpsp
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], 
                                 min_periods = None, center = False, 
                                 win_type = None, on = None, axis = 0).mean()
    
    count_lpsp = lpsp_check[lpsp_check['lpsp'] > instance_data['nse']].count()
    
    #check feasible lpsp
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'

    #create results
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)/
                 (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], 
                                      lpsp['lpsp'])), columns = ['S-','LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, 
                            bplus_df, sminus_df, splus_df, generation_cost], axis=1) 

    time_f = time.time() - time_i
    
    return lcoe_cost, df_results, state, time_f, nsh




def ds_battery_renewable  (solution, demand_df, instance_data, 
                           cost_data, CRF, delta, rand_ob):    
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are batteries and 
    renewable generators (solar panel PV - or Eolic turbine or both), 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    CRF : NUMBER
        percentage to annualize costs
    delta: NUMBER
        percentage - fiscal incentive
    rand_ob : OBJECT OF CLASS RANDOM
        generates random numbers, samples, etc

    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''
    
    #initial parameters 
    time_i = time.time()
    aux_dict = {}
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #battery stated of charge
    soc = {l + '_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l + '_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol}
    #discharge battery
    bminus = {l + '_b-': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    list_ren = [] #renewable generation
    demand_tobe_covered = [] 
    dict_total = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
    cost = {k + '_cost': [0] * LEN_DATA for k in dict_total} #variable cost
    inverter = instance_data['inverter_cost']
    extra_generation = 0  #extra renewavble generation to waste or charge
    nsh = 0 #count not server hours
    
    #calculate cost investment
    for g in solution.generators_dict_sol.values():
        
        lcoe_inf = (g.cost_up * delta + g.cost_r 
                    * delta - g.cost_s + g.cost_fopm) * CRF 
        lcoe_inftot += lcoe_inf 
        list_ren.append(g.id_gen)
    #calculate battery cost investment
    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up * delta + b.cost_r 
                    * delta - b.cost_s + b.cost_fopm) * CRF
        lcoe_inftot += lcoe_inf    
        aux_dict[b.id_bat] = lcoe_inf

    
    sorted_batteries = sorted(aux_dict, key = aux_dict.get, reverse = False) 
    #random order of generators
    rand_ob.create_rand_shuffle(sorted_batteries)
    
    #simulation
    for t in demand_df['t']:
        #initialy demand to be covered is the same that demand 
        demand_tobe_covered = demand_df['demand'][t]
        #calculate soc at 0 time
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = max(b.eb_zero * (1-b.alpha), 0)

        #calculate soc batteries
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = soc[bat + '_soc'][t - 1] * (1 - b.alpha)            

        #calculate renewable generation
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            #calculate cost
            cost[ren + '_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren + '_cost'][t]
            
        #sum all generation
        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] 
                             for i in list_ren)
        ptot += generation_ren
        #surplus energy
        if generation_ren > demand_tobe_covered:
            #supply the load and charge the batteries
            extra_generation = generation_ren - demand_tobe_covered
            #charge batteries until generation extra finish
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]

                if extra_generation > 0:
                    
                    bplus[bat + '_b+'][t] = min(extra_generation, (b.soc_max - soc[bat + '_soc'][t])
                                               /b.efc)
                    #update soc
                    soc[bat + '_soc'][t] += bplus[bat + '_b+'][t] * b.efc
                    extra_generation = extra_generation - bplus[bat + '_b+'][t]

            #if still extra gen is wasted energy   
            splus['s+'][t] = extra_generation
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            demand_tobe_covered = 0
            splustot += costsplus['cost_s+'][t]                    
        else:
            #update demand to be covered
            demand_tobe_covered = demand_tobe_covered - generation_ren    
            
            #supply the load with batteries
            for i in sorted_batteries:
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     #battery have energy to supply all the load
                     if ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd >= demand_tobe_covered / bat.efd):
                         bminus[i + '_b-'][t] = demand_tobe_covered
                         #variable cost
                         cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                         costvopm += cost[i + '_cost'][t]
                         soc[i + '_soc'][t] -= demand_tobe_covered / bat.efd
                         ptot += bminus[i + '_b-'][t]
                         demand_tobe_covered = 0
                     #battery supplied according the most that can give to the load
                     elif ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd > 0):
                        bminus[i + '_b-'][t] = (soc[i + '_soc'][t] - bat.soc_min) * bat.efd
                        #variable cost
                        cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                        costvopm += cost[i + '_cost'][t]
                        ptot += bminus[i + '_b-'][t]
                        soc[i + '_soc'][t] -= bminus[i + '_b-'][t] / bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i + '_b-'][t]
 
        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t]  / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t]               
                
    #mean average lpsp
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], 
                                 min_periods = None, center = False, 
                                 win_type = None, on = None, axis = 0).mean()
    
    count_lpsp = lpsp_check[lpsp_check['lpsp'] > instance_data['nse']].count()
    
    #check feasible lpsp
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'

    #create results
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)/
                 (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], 
                                      lpsp['lpsp'])), columns = ['S-','LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, 
                            bplus_df, sminus_df, splus_df, generation_cost], axis = 1) 

    time_f = time.time() - time_i
    
    return lcoe_cost, df_results, state, time_f, nsh



def ds_dies_batt_renew(solution, demand_df, instance_data, 
                       cost_data, CRF, delta, rand_ob):
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are batteries, diesel generators and 
    renewable generators (solar panel PV - or Eolic turbine or both), 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    CRF : NUMBER
        percentage to annualize costs
    delta: NUMBER
        percentage - fiscal incentive
    rand_ob : OBJECT OF CLASS RANDOM
        generates random numbers, samples, etc

    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''
    #initial parameters 
    time_i = time.time()
    aux_bat = {}
    aux_gen = {}
    lcoe_op = 0 #operative cost
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol}
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #battery stated of charge
    soc = {l + '_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l + '_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l + '_b-': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    list_ren = [] #renewable generation
    demand_tobe_covered = [] 
    dict_total = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
    cost = {k + '_cost': [0] * LEN_DATA for k in dict_total} #variable cost
    extra_generation = 0  #extra renewavble generation to waste or charge
    fuel_cost = instance_data['fuel_cost']
    inverter = instance_data['inverter_cost']
    aux_demand = 0
    AVERAGE_DEMAND = np.mean(demand_df['demand'])
    nsh = 0 #count not server hours

    #calculate investment cost
    for g in solution.generators_dict_sol.values():
        if (g.tec == 'D'):
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s + g.cost_fopm) * CRF 
            lcoe_inftot += lcoe_inf  
            #assume it produces around the average
            prod = min(AVERAGE_DEMAND, g.DG_max)
            lcoe_op = (g.f0 * g.DG_max + g.f1 * prod) * fuel_cost * LEN_DATA
            aux_gen[g.id_gen] = (prod * LEN_DATA) / (lcoe_inf * CRF + lcoe_op)
        else:
            lcoe_inf = (g.cost_up * delta + g.cost_r 
                        * delta - g.cost_s + g.cost_fopm) * CRF 
            lcoe_inftot += lcoe_inf    
            list_ren.append(g.id_gen)
 
    #initial generator always the best lcoe
    sorted_generators = sorted(aux_gen, key = aux_gen.get, reverse = False) 
    
    #reference to generators renewables
    ref = solution.generators_dict_sol[sorted_generators[0]].DG_min
    
    #calculate batteries cost
    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up * delta + b.cost_r 
                    * delta - b.cost_s + b.cost_fopm) * CRF 
        lcoe_inftot += lcoe_inf   
        aux_bat[b.id_bat] = lcoe_inf
            
    #initial battery alwatys the best lcoe
    sorted_batteries = sorted(aux_bat, key = aux_bat.get, reverse = False) 
    #random order of generators
    rand_ob.create_rand_shuffle(sorted_batteries)
    
    #simulation
    for t in demand_df['t']:
        #initialy demand to be covered is the same that demand
        demand_tobe_covered = demand_df['demand'][t]
        aux_demand = 0
        #soc initial simulation
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = max(b.eb_zero * (1 - b.alpha), 0)

        #state of charge of each battery
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = soc[bat + '_soc'][t - 1] * (1 - b.alpha)            

        #calculate all renewable generation
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            cost[ren + '_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren + '_cost'][t]

        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] 
                             for i in list_ren)
        ptot += generation_ren
        #surplus energy
        if generation_ren > demand_tobe_covered:
            #supply the load and charge batteries in order, until extra generation finish
            extra_generation = generation_ren - demand_tobe_covered
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]

                if extra_generation > 0:
                    bplus[bat + '_b+'][t] = min(extra_generation, (b.soc_max - soc[bat + '_soc'][t])
                                               /b.efc)
                    #update soc
                    soc[bat + '_soc'][t] += bplus[bat + '_b+'][t] * b.efc
                    #update extra generation
                    extra_generation = extra_generation - bplus[bat + '_b+'][t]

            #still extra generation, so surplus energy
            splus['s+'][t] = extra_generation
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            demand_tobe_covered = 0
            splustot += costsplus['cost_s+'][t]
        #not renewable enery
        elif(generation_ren == 0):  
            #diesel generators supply at least the reference
            dem2 = demand_tobe_covered
            demand_tobe_covered = max(0, demand_tobe_covered - ref)
            #charge with the batteries
            for i in sorted_batteries:
                #still energy to supply
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     #battery covered all demand
                     if ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd >= demand_tobe_covered / bat.efd):
                         bminus[i + '_b-'][t] = demand_tobe_covered
                         #variable cost
                         cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                         costvopm += cost[i + '_cost'][t]
                         soc[i + '_soc'][t] -= demand_tobe_covered / bat.efd
                         ptot += bminus[i + '_b-'][t]
                         demand_tobe_covered = 0
                    #battery covered until deep od discharge
                     elif ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd > 0):
                        bminus[i + '_b-'][t] = (soc[i + '_soc'][t] - bat.soc_min) * bat.efd
                        #variable cost
                        cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                        costvopm += cost[i + '_cost'][t]
                        ptot += bminus[i + '_b-'][t]
                        soc[i + '_soc'][t] -= bminus[i + '_b-'][t]/bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i + '_b-'][t]

            #batteries supplies all demand            
            if demand_tobe_covered == 0:
                #use one diesel, the reference and calculate suplus if apply
                ptot += ref
                i = sorted_generators[0]
                gen = solution.generators_dict_sol[i]
                p[i][t] = ref
                cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                costvopm += cost[i + '_cost'][t]
                demand_tobe_covered = 0
                if (ref > dem2):
                    splus['s+'][t] = (ref - dem2)
                    costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                    splustot += costsplus['cost_s+'][t]
                
            #need diesel generator
            else:
                #initial generator supply more than min reference
                demand_tobe_covered = demand_tobe_covered + ref
                for j in sorted_generators:
                     gen = solution.generators_dict_sol[j]
                     #lowest that reference turn off
                     if (demand_tobe_covered < gen.DG_min):
                         p[j][t] = 0
                         cost[j + '_cost'][t] = 0
                     #covered all demand
                     elif (gen.DG_max >= demand_tobe_covered):
                         p[j][t] = demand_tobe_covered
                         ptot += p[j][t]
                         cost[j + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[j][t]) * fuel_cost
                         costvopm += cost[j + '_cost'][t]
                         demand_tobe_covered = 0
                    #covered until max rated capacity
                     else:
                        p[j][t] = gen.DG_max
                        ptot += p[j][t]
                        cost[j + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                        costvopm += cost[j + '_cost'][t]
                        #update demand to be covered
                        demand_tobe_covered = demand_tobe_covered - gen.DG_max

        #not enough renewable energy
        else:
            #check if the batteries can supply all the demand
            for i in sorted_batteries:
                bat = solution.batteries_dict_sol[i]
                aux_demand += (soc[i + '_soc'][t] - bat.soc_min) * bat.efd

            #if batteries can supply the load use renewable generator to supply the load,
            #else first charge battery
            if (aux_demand >= demand_tobe_covered):
                demand_tobe_covered = demand_tobe_covered - generation_ren
                for i in sorted_batteries:
                    #supply load with batteries
                     bat = solution.batteries_dict_sol[i]
                     #battery can supply all demand
                     if ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd >= demand_tobe_covered / bat.efd):
                         bminus[i + '_b-'][t] = demand_tobe_covered
                         #variable cost
                         cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                         costvopm += cost[i + '_cost'][t]
                         soc[i + '_soc'][t] -= demand_tobe_covered / bat.efd
                         ptot += bminus[i + '_b-'][t]
                         demand_tobe_covered = 0
                     #battery supply until dept of discharge
                     elif ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd > 0):
                        bminus[i + '_b-'][t] = (soc[i + '_soc'][t] - bat.soc_min) * bat.efd
                        #variable cost
                        cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                        costvopm += cost[i + '_cost'][t]
                        ptot += bminus[i + '_b-'][t]
                        soc[i + '_soc'][t] -= bminus[i + '_b-'][t] / bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i + '_b-'][t]            

            else:
                #charge the bateries
                for bat in sorted_batteries:
                    b = solution.batteries_dict_sol[bat]
                    if generation_ren > 0:
                        bplus[bat + '_b+'][t] = min(generation_ren, (b.soc_max - soc[bat + '_soc'][t])
                                                   / b.efc)
                        soc[bat + '_soc'][t] += bplus[bat + '_b+'][t] * b.efc
                        #update generation ren
                        generation_ren = generation_ren - bplus[bat + '_b+'][t]                

                #if there is enough, suppply part of the demand
                demand_tobe_covered = demand_tobe_covered - generation_ren
                #activate diesel generators
                for j in sorted_generators:
                     gen = solution.generators_dict_sol[j]
                     #lowest than reference, turn off
                     if (demand_tobe_covered < gen.DG_min):
                         p[j][t] = 0
                         cost[j + '_cost'][t] = 0
                     #the diesel can supply all demand
                     elif (gen.DG_max >= demand_tobe_covered):
                         p[j][t] = demand_tobe_covered
                         ptot += p[j][t]
                         cost[j + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[j][t]) * fuel_cost
                         costvopm += cost[j + '_cost'][t]
                         demand_tobe_covered = 0
                    #supply until maximum rated
                     else:
                        p[j][t] = gen.DG_max
                        ptot += p[j][t]
                        cost[j + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                        costvopm += cost[j + '_cost'][t]
                        demand_tobe_covered = demand_tobe_covered - gen.DG_max

        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 

    #mean average lpsp
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], 
                                 min_periods = None, center = False, 
                                 win_type = None, on = None, axis = 0).mean()
    
    count_lpsp = lpsp_check[lpsp_check['lpsp'] > instance_data['nse']].count()
    
    #check feasible lpsp
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'

    #create results
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)/
                 (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], 
                                      lpsp['lpsp'])), columns = ['S-','LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, 
                            bplus_df, sminus_df, splus_df, generation_cost], axis = 1) 

    time_f = time.time() - time_i

    return lcoe_cost, df_results, state, time_f, nsh
   

def dispatch_strategy(sol_try, demand_df, instance_data, cost_data, CRF, delta, rand_ob):
    '''
    this function determines which dispatch strategy to use according to 
    the generators and batteries, and then with this information simulates
    the model according to the selected strategy, returning the best solution
    with its results and if it is feasible or not

    Parameters
    ----------
    sol_try : OBJECT OF SOLUTION CLASS
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    CRF : NUMBER - PERCENT
    delta : NUMBER - PERCENT
    rand_ob : OBJECT OF RANDOM CLASS

    Returns
    -------
    lcoe_cost : NUMBER - DBL
        LCOE of the solution
    df_results : DATAFRAME
        Dataframe with the hourly solution
    state : STRING
        Feasible or not
    time_f : NUMBER - DBL
        Running time
    nsh : NUMBER - DBL
        Total of not served hour 

    '''
    
    list_ds_diesel = ["diesel"]
    list_ds_diesel_renewable = [
        "diesel - solar","diesel - wind", 
        "diesel - solar - wind"
        ]
    
    list_ds_battery_renewable = [
        "battery - solar","battery - wind",
        "battery - solar - wind"
        ]
    
    list_ds_dies_batt_renew = [
        "battery - diesel - wind","battery - diesel - solar", 
        "battery - diesel - solar - wind"
        ]

    #define which dispatch strategy to use 
    strategy_def = select_strategy(generators_dict = sol_try.generators_dict_sol,
                                   batteries_dict = sol_try.batteries_dict_sol) 
    
    #run the dispatch strategy
    if (strategy_def in list_ds_diesel):
        lcoe_cost, df_results, state, time_f, nsh = ds_diesel(sol_try, 
                                                              demand_df, instance_data, cost_data, CRF)
    elif (strategy_def in list_ds_diesel_renewable):
        lcoe_cost, df_results, state, time_f, nsh = ds_diesel_renewable(sol_try,
                                                                        demand_df, instance_data, cost_data,CRF, delta)
    elif (strategy_def in list_ds_battery_renewable):
        lcoe_cost, df_results, state, time_f, nsh = ds_battery_renewable (sol_try, 
                                                                          demand_df, instance_data, cost_data, CRF, delta, rand_ob)
    elif (strategy_def in list_ds_dies_batt_renew):
        lcoe_cost, df_results, state, time_f, nsh = ds_dies_batt_renew(sol_try, 
                                                                       demand_df, instance_data, cost_data, CRF, delta, rand_ob)
    else:
        #no feasible combination
        state = 'no feasible'
        df_results = []
        time_f = 0
        nsh = math.inf
        lcoe_cost = math.inf


    return lcoe_cost, df_results, state, time_f, nsh



class Results():
    def __init__(self, solution, df_results, lcoe):  
        '''
        Load the results to the object associated to the class results

        Parameters
        ----------
        solution
        df_results
        lcoe

        '''
        self.df_results = df_results 
        # general descriptives of the solution
        self.descriptive = {}
        
        # generators data
        
        generators = {}
        try:
            for k in solution.generators_dict_sol.values():
               generators[k.id_gen] = 1
            self.descriptive['generators'] = generators
 
        except:
            for k in solution.generators_dict_sol.values():
                if df_results[k.id_gen].sum() >= 0:
                    generators[k.id_gen] = 1

            self.descriptive['generators'] = generators
        
        # technologies

        tecno_data = {}
        try:
            for i in solution.techonologies_dict_sol:
               tecno_data[i] = 1

            self.descriptive['technologies'] = tecno_data
        except: 
              pass 
              
        bat_data = {}
        try: 
            for l in solution.batteries_dict_sol.values():
               bat_data[l.id_bat] = 1
            self.descriptive['batteries'] = bat_data

        except:
            for l in solution.batteries_dict_sol.values():
                if df_results[l.id_bat + '_b-'].sum() + df_results[l.id_bat + '_b+'].sum() >= 0:
                    bat_data[l.id_bat] = 1

            self.descriptive['batteries'] = bat_data 
        #auxiliar area, is calculated with another function          
        self.descriptive['area'] = 0
            
        # objective function
        self.descriptive['LCOE'] = lcoe
        
    def generation_graph(self, ini, fin):
        '''
        Create a plotly bars graph

        Parameters
        ----------
        ini : INTEGER
        fin : INTEGER
        
        Time range in which the bar graph will be generated, by default 
        it uses all the data but it can be changed according to 
        the user's interest between the initial and final period.

        Returns
        -------
        plot 
        '''
        df_results = copy.deepcopy(self.df_results.iloc[int(ini):int(fin)])
        bars = []
        for key, value in self.descriptive['generators'].items():
            if value == 1:
                bars.append(go.Bar(name = key, x = df_results.index, y = df_results[key]))

        for key, value in self.descriptive['batteries'].items():
            if value == 1:
                column_name = key + '_b-'
                bars.append(go.Bar(name = key, x = df_results.index, y = df_results[column_name]))
        
        bars.append(go.Bar(name = 'Unsupplied Demand',
                           x = df_results.index, y = df_results['S-']))
        
        plot = go.Figure(data = bars)
        plot.add_trace(go.Scatter(x = df_results.index, y = df_results['demand'],
                    mode = 'lines',
                    name = 'Demand',
                    line = dict(color = 'grey', dash = 'dot')))
        
        df_results['b+'] = 0
        for key, value in self.descriptive['batteries'].items():
            if value == 1:
                column_name = key + '_b+'
                df_results['b+'] += df_results[column_name]

        plot.add_trace(go.Bar(x = df_results.index, y = df_results['b+'],
                              base =- 1 * df_results['b+'],
                              marker_color = 'grey',
                              name = 'Charge'
                              ))
        

        # Change the bar mode
        plot.update_layout(barmode = 'stack')
        
        return plot




'''MULTIYEAR FORMULATION'''




def ds_my_diesel (solution, demand_df, instance_data, cost_data, my_data, ir):
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are only Diesel generators, 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    my_data : DICTIONARY
    ir : NUMBER
        Interest rate

    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''

    #initial parameters 
    time_i = time.time()
    auxiliar_dict_generator = {}
    lcoe_op = 0 #operative cost
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol} 
    #variable cost
    cost = {k + '_cost'  : [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #batteries, 0 in this case, report the same of others strategies
    #battery stated of charge
    soc = {l + '_soc' : [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l + '_b+' : [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l + '_b-' : [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    fuel_cost_i = instance_data['fuel_cost']
    inverter = instance_data['inverter_cost']
    AVERAGE_DEMAND = np.mean(demand_df['demand'])
    nsh = 0 #count not server hours
    
    #calculate investments cost
    for g in solution.generators_dict_sol.values():  
        lcoe_inf = g.cost_up + g.cost_r - g.cost_s + g.cost_fopm
        lcoe_inftot += lcoe_inf
        #assume it produces around the average
        prod = min (AVERAGE_DEMAND, g.DG_max)
        lcoe_op = (g.f0 * g.DG_max + g.f1 * prod) * fuel_cost_i * LEN_DATA
        
        auxiliar_dict_generator[g.id_gen] = (prod * LEN_DATA) / (lcoe_inf + lcoe_op)

    #sort to initialize always with the best lcoe diesel generator
    sorted_generators = sorted(auxiliar_dict_generator,
                               key = auxiliar_dict_generator.get, reverse = False) 
    
    #simulation
    for t in demand_df['t']:
        #get the YEAR
        YEAR = math.floor(t / 8760)
        #apply multiyear tax growth
        fuel_cost = fuel_cost_i * (1 + my_data["fuel_tax"]) ** YEAR
        #initilialy demand to be covered is the same that demand
        demand_tobe_covered = demand_df['demand'][t]
        #supply with each generator
        for i in sorted_generators:
             gen = solution.generators_dict_sol[i]
             dg_max_i = gen.DG_max
             #apply anual degradation to generator
             dg_max = dg_max_i * (1 - my_data["diesel_deg"]) ** YEAR
             #if all demand covered not generate
             if (demand_tobe_covered == 0):
                 p[i][t] = 0
                 cost[i + '_cost'][t] = 0
             #if lowest that reference can generate, calculate splus                
             elif (demand_tobe_covered < gen.DG_min):
                 p[i][t] = gen.DG_min
                 ptot += p[i][t]
                 cost[i + '_cost'][t]= (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                 costvopm += cost[i + '_cost'][t]
                 splus['s+'][t] = gen.DG_min - demand_tobe_covered
                 demand_tobe_covered = 0
                 costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                 splustot += costsplus['cost_s+'][t]
             #greatest rate, supply all the demand
             elif (dg_max >= demand_tobe_covered):
                 p[i][t] = demand_tobe_covered
                 ptot += p[i][t]
                 cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                 costvopm += cost[i + '_cost'][t]
                 demand_tobe_covered = 0
             else:
                #supply until rated capacity
                p[i][t] = dg_max
                ptot += p[i][t]
                cost[i + '_cost'][t] = (gen.f0 + gen.f1)* gen.DG_max * fuel_cost
                costvopm += cost[i + '_cost'][t]
                demand_tobe_covered = demand_tobe_covered - dg_max

        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t]          
    
    #mean average lpsp
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], 
                                 min_periods = None, center = False, win_type = None, on = None, axis = 0).mean()
    
    count_lpsp = lpsp_check[lpsp_check['lpsp'] >= instance_data['nse']].count()
    
    #check feasible lpsp
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'

    #create results
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)*(1+ir)
                 / (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), 
                             columns = ['S-','LPSP'])
    
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df,
                            sminus_df, splus_df, generation_cost], axis = 1) 
    
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f, nsh
        


def ds_my_diesel_renewable(solution, demand_df, instance_data, 
                           cost_data, delta, my_data, ir):
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are Diesel generators and 
    renewable generators (solar panel PV - or Eolic turbine or both), 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    delta: NUMBER
        percentage - fiscal incentive
    my_data : DICTIONARY
    ir : NUMBER
        Interest rate

    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''
    #initial parameters 
    time_i = time.time()
    auxiliar_dict_generator = {}
    lcoe_op = 0 #operative cost
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol} 
    #variable cost
    cost = {k + '_cost'  : [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #batteries, 0 in this case, report the same of others strategies
    #battery stated of charge
    soc = {l+'_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l+'_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l+'_b-': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    fuel_cost_i = instance_data['fuel_cost']
    inverter = instance_data['inverter_cost']
    list_ren = [] #renewable generation
    demand_tobe_covered = [] 
    min_ref = math.inf #initial min reference (diesel ref for renewable)
    AVERAGE_DEMAND = np.mean(demand_df['demand'])
    nsh = 0 #count not server hours
 
    #calculate investment cost    
    for g in solution.generators_dict_sol.values():
        if (g.tec == 'D'):
            lcoe_inf = g.cost_up + g.cost_r - g.cost_s + g.cost_fopm
            lcoe_inftot += lcoe_inf   
            #assume it produces around the average
            prod = min(AVERAGE_DEMAND, g.DG_max)
            lcoe_op = (g.f0 * g.DG_max + g.f1 * prod) * fuel_cost_i * LEN_DATA
            auxiliar_dict_generator[g.id_gen] = (prod * LEN_DATA) / (lcoe_inf + lcoe_op)
            
            #get the lowest reference
            if g.DG_min <= min_ref:
                 min_ref = g.DG_min
        else:
            lcoe_inf = (g.cost_up + g.cost_r) * delta - g.cost_s  + g.cost_fopm
            lcoe_inftot += lcoe_inf    
            list_ren.append(g.id_gen)

    #sorted diesel initial the best lcoe generator
    sorted_generators = sorted(auxiliar_dict_generator, 
                               key = auxiliar_dict_generator.get, reverse = False) 
   
    #reference is the generator diesel in the first position, 
    #it is necessary for the renewable objets
    ref = solution.generators_dict_sol[sorted_generators[0]].DG_min
    
    #simulation
    for t in demand_df['t']:
        #initialy demand to be covered is the same that demand
        demand_tobe_covered = demand_df['demand'][t]
        #get the YEAR
        YEAR = math.floor(t / 8760)
        #apply multiyear tax growth
        fuel_cost = fuel_cost_i * (1 + my_data["fuel_tax"]) ** YEAR
        #check is higher than min ref
        if (demand_tobe_covered >= min_ref):
            #calculate renewable generation
            for ren in list_ren:
                renew = solution.generators_dict_sol[ren]
                p[ren][t] = renew.gen_rule[t]
                #calculate cost
                cost[ren + '_cost'][t] = p[ren][t] * renew.cost_vopm
                costvopm += cost[ren + '_cost'][t]

            #total renewable generation
            generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t]
                                 for i in list_ren)
            
            ptot += generation_ren
            #surplus energy
            if generation_ren > (demand_tobe_covered - ref):
                splus['s+'][t] = (generation_ren + ref - demand_tobe_covered)
                costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                splustot += costsplus['cost_s+'][t]
                demand_tobe_covered = ref
            #still demand to be covered    
            else:
                demand_tobe_covered = demand_tobe_covered - generation_ren
            
            #turn on only one generator at minimun load
            if (demand_tobe_covered <= ref):
                n = sorted_generators[0]
                gen = solution.generators_dict_sol[n]
                p[n][t] = ref
                ptot += p[n][t]
                cost[n + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[n][t]) * fuel_cost
                costvopm += cost[n + '_cost'][t]
                demand_tobe_covered = demand_tobe_covered - ref                
            else:
                #supply the demand with diesel
                for i in sorted_generators:
                     gen = solution.generators_dict_sol[i]
                     dg_max_i = gen.DG_max
                     #apply anual degradation to generator
                     dg_max = dg_max_i * (1 - my_data["diesel_deg"]) ** YEAR
                     #lowest that reference, don't turn on
                     if (demand_tobe_covered < gen.DG_min):
                         p[i][t] = 0
                         cost[i + '_cost'][t] = 0
                     #rated capacity is higher than demand, then  supply all demand
                     elif (dg_max >= demand_tobe_covered):
                         p[i][t] = demand_tobe_covered
                         ptot += p[i][t]
                         cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                         costvopm += cost[i + '_cost'][t]
                         demand_tobe_covered = 0
                    #suply until rated capacity
                     else:
                        p[i][t] = dg_max
                        ptot += p[i][t]
                        cost[i + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                        costvopm += cost[i + '_cost'][t]
                        #update demand to be covered
                        demand_tobe_covered = demand_tobe_covered - dg_max

        else:
        #use one diesel
            i = sorted_generators[0]
            gen = solution.generators_dict_sol[i]
            p[i][t] = gen.DG_min
            ptot += p[i][t]
            cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
            costvopm += cost[i + '_cost'][t]
            splus['s+'][t] = (gen.DG_min - demand_tobe_covered)
            demand_tobe_covered = 0
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            splustot += costsplus['cost_s+'][t]
                                  
        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 
      
    #movil average lpsp        
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], min_periods = None,
                                 center = False, win_type = None, on = None, axis = 0).mean()
    count_lpsp = lpsp_check[lpsp_check['lpsp'] >= instance_data['nse']].count()
    
    #check feasible - ,eet the lpsp 
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'

    #create results df
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)*(1+ir)
                 / (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp']))
                             ,columns = ['S-','LPSP'])
    
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, 
                            sminus_df, splus_df, generation_cost], axis = 1) 
    
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f, nsh 


def ds_my_battery_renewable  (solution, demand_df, instance_data,
                              cost_data, delta, rand_ob, my_data, ir):
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are batteries and 
    renewable generators (solar panel PV - or Eolic turbine or both), 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    delta: NUMBER
        percentage - fiscal incentive
    rand_ob : OBJECT OF CLASS RANDOM
        generates random numbers, samples, etc
    my_data : DICTIONARY
    ir : NUMBER
        Interest rate

    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''
    #initial parameters 
    time_i = time.time()
    auxiliar_dict_batteries = {}
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #battery stated of charge
    soc = {l + '_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #charge battery
    bplus = {l + '_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l + '_b-': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    list_ren = [] #renewable generation
    demand_tobe_covered = [] 
    dict_total = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
    cost = {k + '_cost': [0] * LEN_DATA for k in dict_total} #variable cost
    inverter = instance_data['inverter_cost']
    extra_generation = 0  #extra renewaable generation to waste or charge bat
    nsh = 0 #count not server hours
    
    #calculate cost investment
    for g in solution.generators_dict_sol.values():
        lcoe_inf = (g.cost_up + g.cost_r ) * delta - g.cost_s + g.cost_fopm
        lcoe_inftot += lcoe_inf 
        list_ren.append(g.id_gen)

    #calculate battery cost investment
    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up + b.cost_r) * delta - b.cost_s + b.cost_fopm
        lcoe_inftot += lcoe_inf    
        auxiliar_dict_batteries[b.id_bat] = lcoe_inf
    
    sorted_batteries = sorted(auxiliar_dict_batteries, 
                              key = auxiliar_dict_batteries.get, reverse = False) 
    #random order of generators
    rand_ob.create_rand_shuffle(sorted_batteries)
    
    #simulation
    for t in demand_df['t']:
        YEAR = math.floor(t / 8760)
        #initialy demand to be covered is the same that demand 
        demand_tobe_covered = demand_df['demand'][t]
        #calculate soc at 0 time
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = max(b.eb_zero * (1 - b.alpha), 0)
        #calculate soc batteries
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = soc[bat + '_soc'][t - 1] * (1 - b.alpha)            

        #calculate renewable generation
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            #calculate cost
            cost[ren +'_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren + '_cost'][t]

        #sum all generation
        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] 
                             for i in list_ren)
        
        ptot += generation_ren
        #surplus energy
        if generation_ren > demand_tobe_covered:
            #supply the load and charge the batteries
            extra_generation = generation_ren - demand_tobe_covered
            #charge batteries until generation extra finish
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                if extra_generation > 0:
                    #apply anual degradation to battery
                    bt_max_i = b.soc_max
                    bt_max = bt_max_i * (1 - my_data["bat_deg"]) ** YEAR
                    #calculate max allowed energy to load the battery
                    bplus[bat + '_b+'][t] = min(extra_generation, (bt_max - soc[bat + '_soc'][t]) / b.efc)
                    #update soc
                    soc[bat + '_soc'][t] += bplus[bat + '_b+'][t] * b.efc
                    extra_generation = extra_generation - bplus[bat + '_b+'][t]

            #if still extra gen is wasted energy   
            splus['s+'][t] = extra_generation
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            demand_tobe_covered = 0
            splustot += costsplus['cost_s+'][t]                    
        else:
            #update demand to be covered
            demand_tobe_covered = demand_tobe_covered - generation_ren               
            #supply the load with batteries
            for i in sorted_batteries:
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     #battery have energy to supply all the load
                     if ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd >= demand_tobe_covered / bat.efd):
                         bminus[i + '_b-'][t] = demand_tobe_covered
                         cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                         costvopm += cost[i + '_cost'][t]
                         soc[i + '_soc'][t] -= demand_tobe_covered / bat.efd
                         ptot += bminus[i + '_b-'][t]
                         demand_tobe_covered = 0
                     #battery supplied according the most that can give to the load
                     elif ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd > 0):
                        bminus[i + '_b-'][t] = (soc[i + '_soc'][t] - bat.soc_min) * bat.efd
                        cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                        costvopm += cost[i + '_cost'][t]
                        ptot += bminus[i + '_b-'][t]
                        soc[i + '_soc'][t] -= bminus[i + '_b-'][t] / bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i + '_b-'][t]

        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 

    #movil average lpsp        
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], min_periods = None,
                                 center = False, win_type = None, on = None, axis = 0).mean()
    
    count_lpsp = lpsp_check[lpsp_check['lpsp'] >= instance_data['nse']].count()
    
    #meet lpsp - feasible solution
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'
    
    #calculate results
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)*(1+ir)
                 / (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), 
                             columns = ['S-','LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, 
                            sminus_df, splus_df, generation_cost], axis = 1) 
    
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f, nsh


def ds_my_dies_batt_renew(solution, demand_df, instance_data, 
                          cost_data, delta, rand_ob, my_data, ir):
    '''
    This function calculates the dispatch strategy according to the 
    "load-following" rules, when there are batteries, diesel generators and 
    renewable generators (solar panel PV - or Eolic turbine or both), 
    it receives the necessary input parameters and returns the lcoe 
    of the configuration, the dataframe of the results as generation, 
    energy not supplied and surplus, demand.
    Whether or not it is optimal, the execution time and
    the number of hours not served

    Source:  https://bibliotecadigital.udea.edu.co/
    bitstream/10495/14038/1/ArraezOswaldo_2019_MethodologySizingHybrid.pdf

    Parameters
    ----------
    solution : OBJECT OF CLASS SOLUTION
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    delta: NUMBER
        percentage - fiscal incentive
    rand_ob : OBJECT OF CLASS RANDOM
        generates random numbers, samples, etc
    my_data : DICTIONARY
    ir : NUMBER
        Interest rate
        
    Returns
    -------
    lcoe_cost : NUMBER
        lcoe of the configuration
    df_results : DATAFRAME
        results as generation, energy not supplied and surplus, demand.
    state : STRING
        Feasible or not
    time_f : NUMBER
        Total function time.
    nsh : NUMBER
        Total not served hours

    '''
    #initial parameters 
    time_i = time.time()
    auxiliar_dict_batteries = {}
    auxiliar_dict_generator = {}
    lcoe_op = 0 #operative cost
    lcoe_inf = 0 #investment cost
    lcoe_inftot = 0 #total investment cost
    LEN_DATA =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0] * LEN_DATA} #not supplied cost
    costsplus = {'cost_s+': [0] * LEN_DATA} #wasted cost
    splus = {'s+': [0] * LEN_DATA} #wasted energy
    sminus = {'s-': [0] * LEN_DATA} #not supplied energy
    lpsp = {'lpsp': [0] * LEN_DATA} #lpsp calculate
    #generation by each generator
    p = {k: [0] * LEN_DATA for k in solution.generators_dict_sol} 
    ptot = 0 #total generation
    costvopm = 0 #variable cost
    splustot = 0 #wasted energy cost
    sminustot = 0 #not supplied load cost
    #battery stated of charge
    soc = {l + '_soc': [0] * LEN_DATA for l in solution.batteries_dict_sol}
    #charge battery
    bplus = {l + '_b+': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    #discharge battery
    bminus = {l + '_b-': [0] * LEN_DATA for l in solution.batteries_dict_sol} 
    list_ren = [] #renewable generation
    demand_tobe_covered = [] 
    dict_total = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
    cost = {k + '_cost': [0] * LEN_DATA for k in dict_total} #variable cost
    extra_generation = 0  #extra renewavble generation to waste or charge the bat
    fuel_cost_i = instance_data['fuel_cost'] 
    inverter = instance_data['inverter_cost']
    aux_demand = 0
    AVERAGE_DEMAND = np.mean(demand_df['demand'])
    nsh = 0 #count not server hours

    #calculate investment cost
    for g in solution.generators_dict_sol.values():
        if (g.tec == 'D'):
            lcoe_inf = g.cost_up + g.cost_r - g.cost_s + g.cost_fopm
            lcoe_inftot += lcoe_inf  
            #assume it produces around the average
            prod = min (AVERAGE_DEMAND, g.DG_max)
            lcoe_op = (g.f0 * g.DG_max + g.f1 * prod) * fuel_cost_i * LEN_DATA
            auxiliar_dict_generator[g.id_gen] = (prod * LEN_DATA) / (lcoe_inf + lcoe_op)
        else:
            lcoe_inf = (g.cost_up + g.cost_r) * delta - g.cost_s + g.cost_fopm
            lcoe_inftot += lcoe_inf    
            list_ren.append(g.id_gen)

    #initial generator always the best lcoe
    sorted_generators = sorted(auxiliar_dict_generator, 
                               key = auxiliar_dict_generator.get, reverse = False)     
    #reference to generators renewables
    ref = solution.generators_dict_sol[sorted_generators[0]].DG_min
    #calculate batteries cost
    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up + b.cost_r) * delta - b.cost_s + b.cost_fopm
        lcoe_inftot += lcoe_inf   
        auxiliar_dict_batteries[b.id_bat] = lcoe_inf        
    
    #initial battery alwatys the best lcoe
    sorted_batteries = sorted(auxiliar_dict_batteries, 
                              key = auxiliar_dict_batteries.get, reverse = False) 
    #random order of generators
    rand_ob.create_rand_shuffle(sorted_batteries)
    
    #simulation
    for t in demand_df['t']:
        #initialy demand to be covered is the same that demand
        demand_tobe_covered = demand_df['demand'][t]
        aux_demand = 0
        #get the YEAR
        YEAR = math.floor(t / 8760)
        #apply multiyear tax growth
        fuel_cost = fuel_cost_i * (1 + my_data["fuel_tax"]) ** YEAR
        #soc initial simulation
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = max(b.eb_zero * (1 - b.alpha), 0)

        #state of charge of each battery
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat + '_soc'][t] = soc[bat + '_soc'][t - 1] * (1 - b.alpha)            

        #calculate all renewable generation
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            cost[ren + '_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren + '_cost'][t]

        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] 
                             for i in list_ren)
        
        ptot += generation_ren
        #surplus energy
        if generation_ren > demand_tobe_covered:
            #supply the load and charge batteries in order, until extra generation finish
            extra_generation = generation_ren - demand_tobe_covered
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                if extra_generation > 0:
                    bt_max_i = b.soc_max
                    bt_max = bt_max_i * (1 - my_data["bat_deg"]) ** YEAR
                    #calculate max allowed energy to load the battery
                    bplus[bat + '_b+'][t] = min(extra_generation, (bt_max - soc[bat + '_soc'][t]) / b.efc)
                    #update soc
                    soc[bat + '_soc'][t] += bplus[bat + '_b+'][t] * b.efc
                    #update extra generation
                    extra_generation = extra_generation - bplus[bat + '_b+'][t]

            #still extra generation, so surplus energy
            splus['s+'][t] = extra_generation
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            demand_tobe_covered = 0
            splustot += costsplus['cost_s+'][t]

        #not renewable enery
        elif(generation_ren == 0):  
            #diesel generators supply at least the reference
            dem2 = demand_tobe_covered
            demand_tobe_covered = max(0, demand_tobe_covered - ref)
            #charge with the batteries
            for i in sorted_batteries:
                #still energy to supply
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     #battery covered all demand
                     if ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd >= demand_tobe_covered / bat.efd):
                         bminus[i + '_b-'][t] = demand_tobe_covered
                         cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                         costvopm += cost[i + '_cost'][t]
                         soc[i + '_soc'][t] -= demand_tobe_covered / bat.efd
                         ptot += bminus[i + '_b-'][t]
                         demand_tobe_covered = 0
                    #battery covered until deep od discharge
                     elif ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd > 0):
                        bminus[i + '_b-'][t] = (soc[i + '_soc'][t] - bat.soc_min) * bat.efd
                        #variable cost
                        cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                        costvopm += cost[i + '_cost'][t]
                        ptot += bminus[i + '_b-'][t]
                        soc[i + '_soc'][t] -= bminus[i + '_b-'][t] / bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i + '_b-'][t]

            #batteries supplies all demand            
            if demand_tobe_covered == 0:
                #use one diesel, the reference and calculate suplus if apply
                ptot += ref
                i = sorted_generators[0]
                gen = solution.generators_dict_sol[i]
                p[i][t] = ref
                cost[i + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t]) * fuel_cost
                costvopm += cost[i + '_cost'][t]
                demand_tobe_covered = 0
                if (ref > dem2):
                    splus['s+'][t] = (ref - dem2)
                    costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                    splustot += costsplus['cost_s+'][t]
                
            #need diesel generator
            else:
                #initial generator supply more than min reference
                demand_tobe_covered = demand_tobe_covered + ref
                for j in sorted_generators:
                     gen = solution.generators_dict_sol[j]
                     dg_max_i = gen.DG_max
                     #apply anual degradation to generator
                     dg_max = dg_max_i * (1 - my_data["diesel_deg"]) ** YEAR
                     #lowest that reference turn off
                     if (demand_tobe_covered < gen.DG_min):
                         p[j][t] = 0
                         cost[j + '_cost'][t] = 0
                     #covered all demand
                     elif (dg_max >= demand_tobe_covered):
                         p[j][t] = demand_tobe_covered
                         ptot += p[j][t]
                         cost[j + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[j][t]) * fuel_cost
                         costvopm += cost[j + '_cost'][t]
                         demand_tobe_covered = 0
                    #covered until max rated capacity
                     else:
                        p[j][t] = dg_max
                        ptot += p[j][t]
                        cost[j + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                        costvopm += cost[j + '_cost'][t]
                        #update demand to be covered
                        demand_tobe_covered = demand_tobe_covered - dg_max

        #not enough renewable energy
        else:
            #check if the batteries can supply all the demand
            for i in sorted_batteries:
                bat = solution.batteries_dict_sol[i]
                aux_demand += (soc[i + '_soc'][t] - bat.soc_min) * bat.efd

            #if batteries can supply the load use renewable generator 
            #to supply the load, else first charge battery
            if (aux_demand >= demand_tobe_covered):
                demand_tobe_covered = demand_tobe_covered - generation_ren
                for i in sorted_batteries:
                    #supply load with batteries
                     bat = solution.batteries_dict_sol[i]
                     #battery can supply all demand
                     if ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd >= demand_tobe_covered / bat.efd):
                         bminus[i + '_b-'][t] = demand_tobe_covered
                         soc[i + '_soc'][t] -= demand_tobe_covered / bat.efd
                         #variable cost
                         cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                         costvopm += cost[i + '_cost'][t]
                         ptot += bminus[i + '_b-'][t]
                         demand_tobe_covered = 0
                     #battery supply until dept of discharge
                     elif ((soc[i + '_soc'][t] - bat.soc_min) * bat.efd > 0):
                        bminus[i + '_b-'][t] = (soc[i + '_soc'][t] - bat.soc_min) * bat.efd
                        #variable cost
                        cost[i + '_cost'][t] = bminus[i + '_b-'][t] * bat.cost_vopm
                        costvopm += cost[i + '_cost'][t]
                        ptot += bminus[i + '_b-'][t]
                        soc[i + '_soc'][t] -= bminus[i + '_b-'][t] / bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i + '_b-'][t]            
            else:
                #charge the bateries
                for bat in sorted_batteries:
                    b = solution.batteries_dict_sol[bat]
                    if generation_ren > 0:
                        #calculate degradarion battery
                        bt_max_i = b.soc_max
                        bt_max = bt_max_i * (1 - my_data["bat_deg"]) ** YEAR
                        #calculate max allowed energy to load the battery
                        bplus[bat + '_b+'][t] = min(generation_ren, (bt_max - soc[bat + '_soc'][t]) / b.efc)
                        soc[bat + '_soc'][t] += bplus[bat + '_b+'][t] * b.efc
                        #update generation ren
                        generation_ren = generation_ren - bplus[bat + '_b+'][t]                

                #if there is enough, suppply part of the demand
                demand_tobe_covered = demand_tobe_covered - generation_ren
                #activate diesel generators
                for j in sorted_generators:
                     gen = solution.generators_dict_sol[j]
                     dg_max_i = gen.DG_max
                     #apply anual degradation to generator
                     dg_max = dg_max_i * (1 - my_data["diesel_deg"]) ** YEAR
                     #lowest than reference, turn off
                     if (demand_tobe_covered < gen.DG_min):
                         p[j][t] = 0
                         cost[j + '_cost'][t] = 0
                     #the diesel can supply all demand
                     elif (dg_max >= demand_tobe_covered):
                         p[j][t] = demand_tobe_covered
                         ptot += p[j][t]
                         cost[j + '_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[j][t]) * fuel_cost
                         costvopm += cost[j + '_cost'][t]
                         demand_tobe_covered = 0
                    #supply until maximum rated
                     else:
                        p[j][t] = dg_max
                        ptot += p[j][t]
                        cost[j + '_cost'][t] = (gen.f0 + gen.f1) * gen.DG_max * fuel_cost
                        costvopm += cost[j+'_cost'][t]
                        demand_tobe_covered = demand_tobe_covered - dg_max

        #the generators finish, if there is still nse, lpsp is calculated
        if (demand_tobe_covered > 0):
            nsh += 1
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t] / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][1] * sminus['s-'][t]
            elif (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][1] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t]                
                
    #movil average                
    lpsp_df = pd.DataFrame(lpsp['lpsp'], columns = ['lpsp'])
    lpsp_check = lpsp_df.rolling(instance_data['tlpsp'], min_periods = None, 
                                 center = False, win_type = None, on = None, axis = 0).mean()
    count_lpsp = lpsp_check[lpsp_check['lpsp'] >= instance_data['nse']].count()
    
    #calculate feasible - meet lpsp
    if (count_lpsp['lpsp'] > 0):
        state = 'no feasible'
    else:
        state = 'optimal'
    solution.feasible = state

    #create df results
    lcoe_cost = ((sminustot + splustot + lcoe_inftot + costvopm + inverter)*(1+ir)
                 / (sum(demand_df['demand']) - sum(sminus['s-'])))
    
    demand = pd.DataFrame(demand_df['demand'], columns = ['demand'])
    generation = pd.DataFrame(p, columns = [*p.keys()])
    soc_df = pd.DataFrame(soc, columns = [*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns = [*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns = [*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns = [*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), 
                             columns = ['S-','LPSP'])
    
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df,
                            sminus_df, splus_df, generation_cost], axis = 1) 
    
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f, nsh

    
def dispatch_my_strategy(sol_try, demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir):
    '''
    this function determines which dispatch strategy to use according to 
    the generators and batteries, and then with this information simulates
    the model according to the selected strategy, returning the best solution
    with its results and if it is feasible or not

    Parameters
    ----------
    sol_try : OBJECT OF SOLUTION CLASS
    demand_df : DATAFRAME
    instance_data : DICTIONARY
    cost_data : DICTIONARY
    delta : NUMBER - PERCENT
    rand_ob : OBJECT OF RANDOM CLASS
    my_data : DICTIONARY
    ir : NUMBER - PERCENT

    Returns
    -------
    lcoe_cost : NUMBER - DBL
        LCOE of the solution
    df_results : DATAFRAME
        Dataframe with the hourly solution
    state : STRING
        Feasible or not
    time_f : NUMBER - DBL
        Running time
    nsh : NUMBER - DBL
        Total of not served hour 

    '''
    list_ds_diesel = ["diesel"]
    list_ds_diesel_renewable = [
        "diesel - solar","diesel - wind", 
        "diesel - solar - wind"
        ]
    
    list_ds_battery_renewable = [
        "battery - solar","battery - wind",
        "battery - solar - wind"
        ]
    
    list_ds_dies_batt_renew = [
        "battery - diesel - wind","battery - diesel - solar", 
        "battery - diesel - solar - wind"
        ]

    #define which dispatch strategy to use 
    strategy_def = select_strategy(generators_dict = sol_try.generators_dict_sol,
                                   batteries_dict = sol_try.batteries_dict_sol) 
    

    #run the dispatch strategy
    if (strategy_def in list_ds_diesel):
        lcoe_cost, df_results, state, time_f, nsh = ds_my_diesel(sol_try, demand_df, 
                                                                 instance_data, cost_data, my_data, ir)
        
    elif (strategy_def in list_ds_diesel_renewable):
        lcoe_cost, df_results, state, time_f, nsh = ds_my_diesel_renewable(sol_try, 
                                                                           demand_df, instance_data, cost_data, delta, my_data, ir)
        
    elif (strategy_def in list_ds_battery_renewable):
        lcoe_cost, df_results, state, time_f, nsh = ds_my_battery_renewable (sol_try, 
                                                                             demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir)
        
    elif (strategy_def in list_ds_dies_batt_renew):
        lcoe_cost, df_results, state, time_f, nsh = ds_my_dies_batt_renew(sol_try, 
                                                                          demand_df, instance_data, cost_data, delta, rand_ob, my_data, ir)
    
    else:
        #no feasible combination
        state = 'no feasible'
        df_results = []
        time_f = 0
        nsh = math.inf
        lcoe_cost = math.inf


    return lcoe_cost, df_results, state, time_f, nsh

class Results_my():
    def __init__(self, solution, df_results, lcoe): 
        '''
        Load the results to the object associated to the class results

        Parameters
        ----------
        solution
        df_results
        lcoe

        '''
        self.df_results = df_results 
        # general descriptives of the solution
        self.descriptive = {}
        # generators data
        generators = {}
        try:
            for k in solution.generators_dict_sol.values():
               generators[k.id_gen] = 1

            self.descriptive['generators'] = generators
        except:
            for k in solution.generators_dict_sol.values():
                if df_results[k.id_gen].sum() >= 0:
                    generators[k.id_gen] = 1

            self.descriptive['generators'] = generators

        # technologies
        tecno_data = {}
        try:
            for i in solution.techonologies_dict_sol:
               tecno_data[i] = 1

            self.descriptive['technologies'] = tecno_data
        except: 
              pass
              
        bat_data = {}
        try: 
            for l in solution.batteries_dict_sol.values():
               bat_data[l.id_bat] = 1

            self.descriptive['batteries'] = bat_data
        except:
            for l in solution.batteries_dict_sol.values():
                if df_results[l.id_bat + '_b-'].sum() + df_results[l.id_bat + '_b+'].sum() >= 0:
                    bat_data[l.id_bat] = 1

            self.descriptive['batteries'] = bat_data 
        #auxiliar area, is calculated with another function 
        self.descriptive['area'] = 0
        # objective function
        self.descriptive['LCOE'] = lcoe
        
        
    def generation_graph(self, ini, fin):
        '''
        Create a plotly bars graph

        Parameters
        ----------
        ini : INTEGER
        fin : INTEGER
        
        Time range in which the bar graph will be generated, by default 
        it uses all the data but it can be changed according to 
        the user's interest between the initial and final period.

        Returns
        -------
        plot 
        '''
        df_results = copy.deepcopy(self.df_results.iloc[int(ini):int(fin)])
        bars = []
        for key, value in self.descriptive['generators'].items():
            if value == 1:
                bars.append(go.Bar(name = key, x = df_results.index, y = df_results[key]))

        for key, value in self.descriptive['batteries'].items():
            if value == 1:
                column_name = key + '_b-'
                bars.append(go.Bar(name = key, x = df_results.index, y = df_results[column_name]))
            
        bars.append(go.Bar(name = 'Unsupplied Demand'
                           ,x = df_results.index, y = df_results['S-']))
        
        plot = go.Figure(data = bars)
        plot.add_trace(go.Scatter(x = df_results.index, y = df_results['demand'],
                    mode = 'lines',
                    name = 'Demand',
                    line = dict(color = 'grey', dash = 'dot')))
        
        df_results['b+'] = 0
        for key, value in self.descriptive['batteries'].items():
            if value == 1:
                column_name = key + '_b+'
                df_results['b+'] += df_results[column_name]

        #self.df_results['Battery1_b+']+self.df_results['Battery2_b+']
        plot.add_trace(go.Bar(x = df_results.index, y = df_results['b+'],
                              base =- 1 * df_results['b+'],
                              marker_color = 'grey',
                              name = 'Charge'
                              ))
        
        # Change the bar mode
        plot.update_layout(barmode = 'stack')
                
        return plot



