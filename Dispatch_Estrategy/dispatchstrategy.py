# -*- coding: utf-8 -*-
import numpy as np
import time
import pandas as pd
import math
import copy
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'


def def_strategy (batteries_dict, generators_dict):
    
    d=0
    s=0
    b=0
    w=0    
    dispatch = ""
    for gen in generators_dict.values(): 
        if (gen.tec == 'D'):
            d = 1
        elif (gen.tec == 'S'):
            s=1
        elif (gen.tec == 'W'):
            w = 1
    
    if (batteries_dict != {}):
        b = 1
        
    cont = 0
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

def d (solution, demand_df, instance_data, cost_data, CRF):
    time_i = time.time()
    auxiliar_dict_generator = {}
    lcoe_op = 0
    lcoe_inf = 0
    lcoe_inftot = 0
    len_data =  len(demand_df['demand'])
    costsminus = {'cost_s-': [0]*len_data}
    splus = {'s+': [0]*len_data}
    sminus = {'s-': [0]*len_data}
    lpsp = {'lpsp': [0]*len_data}
    p = {k : [0]*len_data for k in solution.generators_dict_sol}
    cost = {k+'_cost'  : [0]*len_data for k in solution.generators_dict_sol}
    ptot = 0
    costvopm = 0
    splustot = 0
    sminustot = 0
    soc = {l+'_soc' : [0]*len_data for l in solution.batteries_dict_sol}
    bplus = {l+'_b+' : [0]*len_data for l in solution.batteries_dict_sol}
    bminus = {l+'_b-' : [0]*len_data for l in solution.batteries_dict_sol}
    fuel_cost = instance_data['fuel_cost']
    
    for g in solution.generators_dict_sol.values():  
        lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * CRF + g.cost_fopm
        #lcoe_inf = g.cost_up + g.cost_r - g.cost_s + g.cost_fopm
        lcoe_inftot += lcoe_inf
        lcoe_op =  (g.f0 + g.f1)*g.DG_max*fuel_cost * len_data
        auxiliar_dict_generator[g.id_gen] = (g.DG_max * len_data) / (lcoe_inf*CRF + lcoe_op)
           
    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=False) 

    
    for t in demand_df['t']:
        demand_tobe_covered = demand_df['demand'][t]
        for i in sorted_generators:
             gen = solution.generators_dict_sol[i]
             if (demand_tobe_covered < gen.DG_min):
                 p[i][t] = 0
                 cost[i+'_cost'][t]=0
             elif (gen.DG_max >= demand_tobe_covered):
                 p[i][t] = demand_tobe_covered
                 ptot += p[i][t]
                 cost[i+'_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t])*fuel_cost
                 costvopm += cost[i+'_cost'][t]
                 demand_tobe_covered = 0
             else:
                p[i][t] = gen.DG_max
                ptot += p[i][t]
                cost[i+'_cost'][t] = (gen.f0 + gen.f1)* gen.DG_max * fuel_cost
                costvopm += cost[i+'_cost'][t]
                demand_tobe_covered = demand_tobe_covered - gen.DG_max
        if (demand_tobe_covered > 0):
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t]  / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][0] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 
            #costsminus['cost_s-'][t] = sminus['s-'][t] * instance_data["sminus_cost"]           
             

    if (np.mean(lpsp['lpsp']) >= instance_data['nse']):
        state = 'False'
    else:
        state = 'optimal'
    lcoe_cost = sminustot + splustot + (lcoe_inftot + costvopm)/ptot
    demand = pd.DataFrame(demand_df['demand'], columns=['demand'])
    generation = pd.DataFrame(p, columns=[*p.keys()])
    soc_df = pd.DataFrame(soc, columns=[*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns=[*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns=[*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns=[*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), columns = ['S-', 'LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, sminus_df, splus_df, generation_cost], axis=1) 
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f
    
    

def D_plus_S_and_or_W (solution, demand_df, instance_data, cost_data, CRF, delta):
    time_i = time.time()
    auxiliar_dict_generator = {}
    list_ren = []
    demand_tobe_covered = []
    lcoe_op = 0
    lcoe_inf = 0
    lcoe_inftot = 0
    len_data =  len(demand_df['demand'])
    min_ref = math.inf
    p = {k : [0]*len_data for k in solution.generators_dict_sol}
    cost = {k+'_cost'  : [0]*len_data for k in solution.generators_dict_sol}
    fuel_cost = instance_data['fuel_cost']
    costsplus = {'cost_s+': [0]*len_data}
    costsminus = {'cost_s-': [0]*len_data}
    splus = {'s+': [0]*len_data}
    sminus = {'s-': [0]*len_data}
    lpsp = {'lpsp': [0]*len_data}
    soc = {l+'_soc' : [0]*len_data for l in solution.batteries_dict_sol}
    bplus = {l+'_b+' : [0]*len_data for l in solution.batteries_dict_sol}
    bminus = {l+'_b-' : [0]*len_data for l in solution.batteries_dict_sol}
    ptot = 0
    costvopm = 0
    splustot = 0
    sminustot = 0
    
    for g in solution.generators_dict_sol.values():
        if (g.tec == 'D'):
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * CRF + g.cost_fopm
            #lcoe_inf = g.cost_up + g.cost_r - g.cost_s + g.cost_fopm
            lcoe_inftot += lcoe_inf          
            lcoe_op =  (g.f0 + g.f1)*g.DG_max*fuel_cost * len_data
            auxiliar_dict_generator[g.id_gen] = (g.DG_max * len_data) / (lcoe_inf*CRF + lcoe_op)
            if g.DG_min <= min_ref:
                 min_ref = g.DG_min
        else:
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
            #lcoe_inf = (g.cost_up + g.cost_r - g.cost_s)*delta + g.cost_fopm
            lcoe_inftot += lcoe_inf    
            list_ren.append(g.id_gen)

    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=False) 

    
    ref = solution.generators_dict_sol[sorted_generators[0]].DG_min
    
    for t in demand_df['t']:
        demand_tobe_covered = demand_df['demand'][t]
        if (demand_tobe_covered >= min_ref):
            for ren in list_ren:
                renew = solution.generators_dict_sol[ren]
                p[ren][t] = renew.gen_rule[t]
                cost[ren+'_cost'][t] = p[ren][t] * renew.cost_vopm
                costvopm += cost[ren+'_cost'][t]
            generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] for i in list_ren)
            ptot += generation_ren
            if generation_ren > (demand_tobe_covered - ref):
                splus['s+'][t] = (generation_ren + ref - demand_tobe_covered)
                costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                splustot += costsplus['cost_s+'][t]
                demand_tobe_covered = ref
            else:
                demand_tobe_covered = demand_tobe_covered - generation_ren
            
            if (demand_tobe_covered <= ref):
                n = sorted_generators[0]
                gen = solution.generators_dict_sol[n]
                p[n][t] = ref
                ptot += p[n][t]
                cost[n+'_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[n][t])*fuel_cost
                costvopm += cost[n+'_cost'][t]
                demand_tobe_covered = demand_tobe_covered - ref
                
            else:
                
                for i in sorted_generators:
                     gen = solution.generators_dict_sol[i]
                     if (demand_tobe_covered < gen.DG_min):
                         p[i][t] = 0
                         cost[i+'_cost'][t]=0
                     elif (gen.DG_max >= demand_tobe_covered):
                         p[i][t] = demand_tobe_covered
                         ptot += p[i][t]
                         cost[i+'_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[i][t])*fuel_cost
                         costvopm += cost[i+'_cost'][t]
                         demand_tobe_covered = 0
                     else:
                        p[i][t] = gen.DG_max
                        ptot += p[i][t]
                        cost[i+'_cost'][t] = (gen.f0 + gen.f1)* gen.DG_max * fuel_cost
                        costvopm += cost[i+'_cost'][t]
                        demand_tobe_covered = demand_tobe_covered - gen.DG_max
        if (demand_tobe_covered > 0):
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t]  / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][0] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 
            #costsminus['cost_s-'][t] = sminus['s-'][t] * instance_data["sminus_cost"]   
                
    if (np.mean(lpsp['lpsp']) >= instance_data['nse']):
        state = 'False'
    else:
        state = 'optimal'
    lcoe_cost = sminustot + splustot + (lcoe_inftot + costvopm)/ptot
    demand = pd.DataFrame(demand_df['demand'], columns=['demand'])
    generation = pd.DataFrame(p, columns=[*p.keys()])
    soc_df = pd.DataFrame(soc, columns=[*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns=[*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns=[*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns=[*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), columns = ['S-', 'LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, sminus_df, splus_df, generation_cost], axis=1) 
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f 




def B_plus_S_and_or_W  (solution, demand_df, instance_data, cost_data, CRF, delta):
    time_i = time.time()
    auxiliar_dict_batteries = {}
    list_ren = []
    demand_tobe_covered = []
    lcoe_inf = 0
    lcoe_inftot = 0
    len_data =  len(demand_df['demand'])
    dict_total = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
    p = {k : [0]*len_data for k in solution.generators_dict_sol}
    soc = {l+'_soc' : [0]*len_data for l in solution.batteries_dict_sol}
    bplus = {l+'_b+' : [0]*len_data for l in solution.batteries_dict_sol}
    bminus = {l+'_b-' : [0]*len_data for l in solution.batteries_dict_sol}
    cost = {k+'_cost' : [0]*len_data for k in dict_total}
    costsplus = {'cost_s+': [0]*len_data}
    costsminus = {'cost_s-': [0]*len_data}
    splus = {'s+': [0]*len_data}
    sminus = {'s-': [0]*len_data}
    lpsp = {'lpsp': [0]*len_data}
    ptot = 0
    costvopm = 0
    splustot = 0
    sminustot = 0
    extra_generation = 0 
    for g in solution.generators_dict_sol.values():
        lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
        #lcoe_inf = (g.cost_up + g.cost_r - g.cost_s)*delta + g.cost_fopm
        lcoe_inftot += lcoe_inf 
        list_ren.append(g.id_gen)

    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up + b.cost_r - b.cost_s) * delta * CRF + b.cost_fopm
        #lcoe_inf = (b.cost_up + b.cost_r - b.cost_s)*delta + b.cost_fopm
        lcoe_inftot += lcoe_inf    
        auxiliar_dict_batteries[b.id_bat] = (b.soc_max * len_data) / (lcoe_inf*CRF)        
    

    sorted_batteries = sorted(auxiliar_dict_batteries, key=auxiliar_dict_batteries.get,reverse=False) 

    for t in demand_df['t']:
        demand_tobe_covered = demand_df['demand'][t]
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat+'_soc'][t] = max(b.eb_zero * (1-b.alpha),0)
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat+'_soc'][t] = soc[bat+'_soc'][t-1] * (1-b.alpha)            
                
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            cost[ren+'_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren+'_cost'][t]
        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] for i in list_ren)
        ptot += generation_ren
        if generation_ren > demand_tobe_covered:
            extra_generation = generation_ren - demand_tobe_covered
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                if extra_generation > 0:
                    bplus[bat+'_b+'][t] = min(extra_generation,(b.soc_max - soc[bat+'_soc'][t])/b.efc)
                    soc[bat+'_soc'][t] += bplus[bat+'_b+'][t] * b.efc
                    extra_generation = extra_generation - bplus[bat+'_b+'][t]
                
            splus['s+'][t] = extra_generation
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            demand_tobe_covered = 0
            splustot += costsplus['cost_s+'][t]
                    
        else:
            demand_tobe_covered = demand_tobe_covered - generation_ren    
            
            for i in sorted_batteries:
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     if ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd >= demand_tobe_covered/bat.efd):
                         bminus[i+'_b-'][t] = demand_tobe_covered
                         soc[i+'_soc'][t] -= demand_tobe_covered/bat.efd
                         ptot += bminus[i+'_b-'][t]
                         demand_tobe_covered = 0
                     elif ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd > 0):
                        bminus[i+'_b-'][t] = (soc[i+'_soc'][t] - bat.soc_min)*bat.efd
                        ptot += bminus[i+'_b-'][t]
                        soc[i+'_soc'][t] -= bminus[i+'_b-'][t]/bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i+'_b-'][t]
        
        if (demand_tobe_covered > 0):
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t]  / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][0] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 
            #costsminus['cost_s-'][t] = sminus['s-'][t] * instance_data["sminus_cost"]                 
                
                
            
    if (np.mean(lpsp['lpsp']) >= instance_data['nse']):
        state = 'False'
    else:
        state = 'optimal'
    lcoe_cost = sminustot + splustot + (lcoe_inftot + costvopm)/ptot
    demand = pd.DataFrame(demand_df['demand'], columns=['demand'])
    generation = pd.DataFrame(p, columns=[*p.keys()])
    soc_df = pd.DataFrame(soc, columns=[*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns=[*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns=[*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns=[*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), columns = ['S-', 'LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, sminus_df, splus_df, generation_cost], axis=1) 
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f 


def B_plus_S_and_or_W2  (solution, demand_df, instance_data, cost_data, CRF, delta):
    time_i = time.time()
    auxiliar_dict_batteries = {}
    list_ren = []
    demand_tobe_covered = []
    lcoe_inf = 0
    lcoe_inftot = 0
    len_data =  len(demand_df['demand'])
    dict_total = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
    p = {k : [0]*len_data for k in solution.generators_dict_sol}
    soc = {l+'_soc' : [0]*len_data for l in solution.batteries_dict_sol}
    bplus = {l+'_b+' : [0]*len_data for l in solution.batteries_dict_sol}
    bminus = {l+'_b-' : [0]*len_data for l in solution.batteries_dict_sol}
    cost = {k+'_cost' : [0]*len_data for k in dict_total}
    costsplus = {'cost_s+': [0]*len_data}
    costsminus = {'cost_s-': [0]*len_data}
    splus = {'s+': [0]*len_data}
    sminus = {'s-': [0]*len_data}
    lpsp = {'lpsp': [0]*len_data}
    ptot = 0
    costvopm = 0
    splustot = 0
    sminustot = 0
    extra_generation = 0
    demand_battery = {l+'_min' : [0]*len_data for l in solution.batteries_dict_sol}

    for g in solution.generators_dict_sol.values():
        lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
        lcoe_inftot += lcoe_inf 
        list_ren.append(g.id_gen)

    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up + b.cost_r - b.cost_s) * delta * CRF + b.cost_fopm
        lcoe_inftot += lcoe_inf    
        auxiliar_dict_batteries[b.id_bat] = (b.soc_max * len_data) / (lcoe_inf*CRF)        
    
    
    sorted_batteries = sorted(auxiliar_dict_batteries, key=auxiliar_dict_batteries.get,reverse=False) 

    for t in demand_df['t']:
        demand_tobe_covered = demand_df['demand'][t]
        total_demand_batteries = 0
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat+'_soc'][t] = max(b.eb_zero * (1-b.alpha),0)
                if (soc[bat+'_soc'][t] < b.soc_min):
                    demand_battery[bat+'_min'][t] = max((b.soc_min - soc[bat+'_soc'][t])/b.efc,0)
                    total_demand_batteries += demand_battery[bat+'_min'][t]
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat+'_soc'][t] = soc[bat+'_soc'][t-1] * (1-b.alpha)  
                if (soc[bat+'_soc'][t] < b.soc_min):
                    demand_battery[bat+'_min'][t] = max((b.soc_min - soc[bat+'_soc'][t])/b.efc,0)
                    total_demand_batteries += demand_battery[bat+'_min'][t]
                
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            cost[ren+'_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren+'_cost'][t]
        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] for i in list_ren)
        ptot += generation_ren
        if generation_ren >= (demand_tobe_covered + total_demand_batteries):
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                if (demand_battery[bat+'_min'][t] > 0 and generation_ren > 0):
                    bplus[bat+'_b+'][t] += min(generation_ren,demand_battery[bat+'_min'][t])
                    soc[bat+'_soc'][t] += bplus[bat+'_b+'][t] * b.efc
                    generation_ren = generation_ren - bplus[bat+'_b+'][t]
            extra_generation = max(generation_ren - demand_tobe_covered,0)
            for bat in sorted_batteries:
                aux = 0
                b = solution.batteries_dict_sol[bat]
                if extra_generation > 0 and ((b.soc_max - soc[bat+'_soc'][t])/b.efc) > 0:
                    aux = min(extra_generation,(b.soc_max - soc[bat+'_soc'][t])/b.efc)
                    bplus[bat+'_b+'][t] += aux
                    soc[bat+'_soc'][t] += aux * b.efc
                    extra_generation = extra_generation - aux
            if (extra_generation > 0):
                splus['s+'][t] = extra_generation                
                costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
                splustot += costsplus['cost_s+'][t]
            demand_tobe_covered = 0       
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                if (demand_battery[bat+'_min'][t] > 0 and generation_ren > 0):
                    bplus[bat+'_b+'][t] = min(generation_ren,demand_battery[bat+'_min'][t])
                    soc[bat+'_soc'][t] += bplus[bat+'_b+'][t] * b.efc
                    generation_ren = generation_ren - bplus[bat+'_b+'][t]
            
            demand_tobe_covered = demand_tobe_covered - generation_ren    
            
            for i in sorted_batteries:
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     if ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd >= demand_tobe_covered/bat.efd):
                         bminus[i+'_b-'][t] = demand_tobe_covered
                         soc[i+'_soc'][t] -= demand_tobe_covered/bat.efd
                         ptot += bminus[i+'_b-'][t]
                         demand_tobe_covered = 0
                     elif ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd > 0):
                        bminus[i+'_b-'][t] = (soc[i+'_soc'][t] - bat.soc_min)*bat.efd
                        ptot += bminus[i+'_b-'][t]
                        soc[i+'_soc'][t] -= bminus[i+'_b-'][t]/bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i+'_b-'][t]
        
        if (demand_tobe_covered > 0):
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t]  / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][0] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 
            #costsminus['cost_s-'][t] = sminus['s-'][t] * instance_data["sminus_cost"]                 
                
                
        
    if (np.mean(lpsp['lpsp']) >= instance_data['nse']):
        state = 'False'
    else:
        state = 'optimal'
    lcoe_cost = sminustot + splustot + (lcoe_inftot + costvopm)/ptot
    demand = pd.DataFrame(demand_df['demand'], columns=['demand'])
    generation = pd.DataFrame(p, columns=[*p.keys()])
    soc_df = pd.DataFrame(soc, columns=[*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns=[*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns=[*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns=[*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), columns = ['S-', 'LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, sminus_df, splus_df, generation_cost], axis=1) 
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f 



def B_plus_D_plus_Ren(solution, demand_df, instance_data, cost_data, CRF, delta):
    
    fuel_cost = instance_data['fuel_cost']
    time_i = time.time()
    auxiliar_dict_batteries = {}
    auxiliar_dict_generator = {}
    list_ren = []
    demand_tobe_covered = []
    lcoe_inf = 0
    lcoe_inftot = 0
    len_data =  len(demand_df['demand'])
    dict_total = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
    p = {k : [0]*len_data for k in solution.generators_dict_sol}
    soc = {l+'_soc' : [0]*len_data for l in solution.batteries_dict_sol}
    bplus = {l+'_b+' : [0]*len_data for l in solution.batteries_dict_sol}
    bminus = {l+'_b-' : [0]*len_data for l in solution.batteries_dict_sol}
    cost = {k+'_cost' : [0]*len_data for k in dict_total}
    costsplus = {'cost_s+': [0]*len_data}
    costsminus = {'cost_s-': [0]*len_data}
    splus = {'s+': [0]*len_data}
    sminus = {'s-': [0]*len_data}
    lpsp = {'lpsp': [0]*len_data}
    ptot = 0
    costvopm = 0
    splustot = 0
    sminustot = 0
    extra_generation = 0
    aux_demand = 0

    for g in solution.generators_dict_sol.values():
        if (g.tec == 'D'):
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * CRF + g.cost_fopm
            #lcoe_inf = g.cost_up + g.cost_r - g.cost_s + g.cost_fopm
            lcoe_inftot += lcoe_inf          
            lcoe_op =  (g.f0 + g.f1)*g.DG_max*fuel_cost * len_data
            auxiliar_dict_generator[g.id_gen] = (g.DG_max * len_data) / (lcoe_inf*CRF + lcoe_op)
        else:
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
            #lcoe_inf = (g.cost_up + g.cost_r - g.cost_s)*delta + g.cost_fopm
            lcoe_inftot += lcoe_inf    
            list_ren.append(g.id_gen)

    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=False) 
    ref = solution.generators_dict_sol[sorted_generators[0]].DG_min
    
    for b in solution.batteries_dict_sol.values():
        lcoe_inf = (b.cost_up + b.cost_r - b.cost_s) * delta * CRF + b.cost_fopm
        #lcoe_inf = (b.cost_up + b.cost_r - b.cost_s)*delta + b.cost_fopm
        lcoe_inftot += lcoe_inf   
        auxiliar_dict_batteries[b.id_bat] = (b.soc_max * len_data) / (lcoe_inf*CRF)        
    

    sorted_batteries = sorted(auxiliar_dict_batteries, key=auxiliar_dict_batteries.get,reverse=False) 


    for t in demand_df['t']:
        demand_tobe_covered = demand_df['demand'][t]
        aux_demand = 0
        if t == 0:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat+'_soc'][t] = max(b.eb_zero * (1-b.alpha),0)
        else:
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                soc[bat+'_soc'][t] = soc[bat+'_soc'][t-1] * (1-b.alpha)            
                
        for ren in list_ren:
            renew = solution.generators_dict_sol[ren]
            p[ren][t] = renew.gen_rule[t]
            cost[ren+'_cost'][t] = p[ren][t] * renew.cost_vopm
            costvopm += cost[ren+'_cost'][t]
        generation_ren = sum(solution.generators_dict_sol[i].gen_rule[t] for i in list_ren)
        ptot += generation_ren
        if generation_ren > demand_tobe_covered:
            extra_generation = generation_ren - demand_tobe_covered
            for bat in sorted_batteries:
                b = solution.batteries_dict_sol[bat]
                if extra_generation > 0:
                    bplus[bat+'_b+'][t] = min(extra_generation,(b.soc_max - soc[bat+'_soc'][t])/b.efc)
                    soc[bat+'_soc'][t] += bplus[bat+'_b+'][t] * b.efc
                    extra_generation = extra_generation - bplus[bat+'_b+'][t]
                
            splus['s+'][t] = extra_generation
            costsplus['cost_s+'][t] = splus['s+'][t] * instance_data["splus_cost"]
            demand_tobe_covered = 0
            splustot += costsplus['cost_s+'][t]
                    
        elif(generation_ren == 0):  
            demand_tobe_covered = demand_tobe_covered - ref
            p[sorted_generators[0]][t] = ref
            for i in sorted_batteries:
                if demand_tobe_covered > 0:
                     bat = solution.batteries_dict_sol[i]
                     if ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd >= demand_tobe_covered/bat.efd):
                         bminus[i+'_b-'][t] = demand_tobe_covered
                         soc[i+'_soc'][t] -= demand_tobe_covered/bat.efd
                         ptot += bminus[i+'_b-'][t]
                         demand_tobe_covered = 0
                     elif ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd > 0):
                        bminus[i+'_b-'][t] = (soc[i+'_soc'][t] - bat.soc_min)*bat.efd
                        ptot += bminus[i+'_b-'][t]
                        soc[i+'_soc'][t] -= bminus[i+'_b-'][t]/bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i+'_b-'][t]
                        
            if demand_tobe_covered == 0:
                ptot += ref
            else:
                demand_tobe_covered = demand_tobe_covered + ref
                p[sorted_generators[0]][t] = 0
                for j in sorted_generators:
                     gen = solution.generators_dict_sol[j]
                     if (demand_tobe_covered < gen.DG_min):
                         p[j][t] = 0
                         cost[j+'_cost'][t]=0
                     elif (gen.DG_max >= demand_tobe_covered):
                         p[j][t] = demand_tobe_covered
                         ptot += p[j][t]
                         cost[j+'_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[j][t])*fuel_cost
                         costvopm += cost[j+'_cost'][t]
                         demand_tobe_covered = 0
                     else:
                        p[j][t] = gen.DG_max
                        ptot += p[j][t]
                        cost[j+'_cost'][t] = (gen.f0 + gen.f1)* gen.DG_max * fuel_cost
                        costvopm += cost[j+'_cost'][t]
                        demand_tobe_covered = demand_tobe_covered - gen.DG_max
        
        else:
            for i in sorted_batteries:
                bat = solution.batteries_dict_sol[i]
                aux_demand += (soc[i+'_soc'][t] - bat.soc_min)*bat.efd
            
            if (aux_demand >= demand_tobe_covered):
                demand_tobe_covered = demand_tobe_covered - generation_ren
                for i in sorted_batteries:
                     bat = solution.batteries_dict_sol[i]
                     if ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd >= demand_tobe_covered/bat.efd):
                         bminus[i+'_b-'][t] = demand_tobe_covered
                         soc[i+'_soc'][t] -= demand_tobe_covered/bat.efd
                         ptot += bminus[i+'_b-'][t]
                         demand_tobe_covered = 0
                     elif ((soc[i+'_soc'][t] - bat.soc_min)*bat.efd > 0):
                        bminus[i+'_b-'][t] = (soc[i+'_soc'][t] - bat.soc_min)*bat.efd
                        ptot += bminus[i+'_b-'][t]
                        soc[i+'_soc'][t] -= bminus[i+'_b-'][t]/bat.efd
                        demand_tobe_covered = demand_tobe_covered - bminus[i+'_b-'][t]            
            else:
                for bat in sorted_batteries:
                    b = solution.batteries_dict_sol[bat]
                    if generation_ren > 0:
                        bplus[bat+'_b+'][t] = min(generation_ren,(b.soc_max - soc[bat+'_soc'][t])/b.efc)
                        soc[bat+'_soc'][t] += bplus[bat+'_b+'][t] * b.efc
                        generation_ren = generation_ren - bplus[bat+'_b+'][t]                
                
                demand_tobe_covered = demand_tobe_covered - generation_ren
                for j in sorted_generators:
                     gen = solution.generators_dict_sol[j]
                     if (demand_tobe_covered < gen.DG_min):
                         p[j][t] = 0
                         cost[j+'_cost'][t]=0
                     elif (gen.DG_max >= demand_tobe_covered):
                         p[j][t] = demand_tobe_covered
                         ptot += p[j][t]
                         cost[j+'_cost'][t] = (gen.f0 * gen.DG_max + gen.f1 * p[j][t])*fuel_cost
                         costvopm += cost[j+'_cost'][t]
                         demand_tobe_covered = 0
                     else:
                        p[j][t] = gen.DG_max
                        ptot += p[j][t]
                        cost[j+'_cost'][t] = (gen.f0 + gen.f1)* gen.DG_max * fuel_cost
                        costvopm += cost[j+'_cost'][t]
                        demand_tobe_covered = demand_tobe_covered - gen.DG_max

        
        if (demand_tobe_covered > 0):
            sminus['s-'][t] = demand_tobe_covered
            lpsp['lpsp'][t] = sminus['s-'][t]  / demand_df['demand'][t]
            if (lpsp['lpsp'][t] <= cost_data['NSE_COST']["L1"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L1"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L2"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L2"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L3"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L3"][0] * sminus['s-'][t]
            elif (lpsp['lpsp'][t]  <= cost_data['NSE_COST']["L4"][0]):
                costsminus['cost_s-'][t] = cost_data['NSE_COST']["L4"][0] * sminus['s-'][t]
            
            sminustot += costsminus['cost_s-'][t] 
            #costsminus['cost_s-'][t] = sminus['s-'][t] * instance_data["sminus_cost"]                 
                
                      
    if (np.mean(lpsp['lpsp']) >= instance_data['nse']):
        state = 'False'
    else:
        state = 'optimal'
    solution.feasible = state

    lcoe_cost = sminustot + splustot + (lcoe_inftot + costvopm)/ptot
    demand = pd.DataFrame(demand_df['demand'], columns=['demand'])
    generation = pd.DataFrame(p, columns=[*p.keys()])
    soc_df = pd.DataFrame(soc, columns=[*soc.keys()])
    bplus_df = pd.DataFrame(bplus, columns=[*bplus.keys()])
    bminus_df = pd.DataFrame(bminus, columns=[*bminus.keys()])    
    generation_cost = pd.DataFrame(cost, columns=[*cost.keys()])
    sminus_df = pd.DataFrame(list(zip(sminus['s-'], lpsp['lpsp'])), columns = ['S-', 'LPSP'])
    splus_df = pd.DataFrame(splus['s+'], columns = ['Wasted Energy'])
    df_results = pd.concat([demand, generation, bminus_df, soc_df, bplus_df, sminus_df, splus_df, generation_cost], axis=1) 
    time_f = time.time() - time_i
    return lcoe_cost, df_results, state, time_f 
   

class Results():
    def __init__(self, solution, df_results, lcoe):
        
        self.df_results = df_results 
        
        # general descriptives of the solution
        self.descriptive = {}
        
        # generators 
        generators = {}
        try:
            for k in solution.generators_dict_sol.values():
               generators[k.id_gen] = 1
            self.descriptive['generators'] = generators
        except:
            for k in solution.generators_dict_sol.values():
                if df_results[k.id_gen].sum() > 0:
                    generators[k.id_gen] = 1
            self.descriptive['generators'] = generators
        # technologies
        tecno_data = {}
        try:
            for i in solution.techonologies_dict_sol:
               tecno_data[i] = 1
            self.descriptive['technologies'] = tecno_data
        except: #TODO
              a=1 
              
        bat_data = {}
        try: 
            for l in solution.batteries_dict_sol.values():
               bat_data[l.id_bat] = 1
            self.descriptive['batteries'] = bat_data
        except:
            for l in solution.batteries_dict_sol.values():
                if df_results[l.id_bat+'_b-'].sum() + df_results[l.id_bat+'_b+'].sum() > 0:
                    bat_data[l.id_bat] = 1
            self.descriptive['batteries'] = bat_data 
                  

        self.descriptive['area'] = 0
            
        # objective function
        self.descriptive['LCOE'] = lcoe
        
        
        
    def generation_graph(self,ini,fin):
        df_results = copy.deepcopy(self.df_results.iloc[int(ini):int(fin)])
        bars = []
        for key, value in self.descriptive['generators'].items():
            if value==1:
                bars.append(go.Bar(name=key, x=df_results.index, y=df_results[key]))
        for key, value in self.descriptive['batteries'].items():
            if value==1:
                column_name = key+'_b-'
                bars.append(go.Bar(name=key, x=df_results.index, y=df_results[column_name]))
  
        bars.append(go.Bar(name='Unsupplied Demand',x=df_results.index, y=df_results['S-']))
                
        plot = go.Figure(data=bars)
        
        
        plot.add_trace(go.Scatter(x=df_results.index, y=df_results['demand'],
                    mode='lines',
                    name='Demand',
                    line=dict(color='grey', dash='dot')))
        
        df_results['b+'] = 0
        for key, value in self.descriptive['batteries'].items():
            if value==1:
                column_name = key+'_b+'
                df_results['b+'] += df_results[column_name]
        #self.df_results['Battery1_b+']+self.df_results['Battery2_b+']
        plot.add_trace(go.Bar(x=df_results.index, y=df_results['b+'],
                              base=-1*df_results['b+'],
                              marker_color='grey',
                              name='Charge'
                              ))
        
        # Set values y axis
        #plot.update_yaxes(range=[-self.df_results['b+'].max()-50, self.df_results['demand'].max()+200])
        #plot.update_yaxes(range=[-10, 30])
        # Change the bar mode
        plot.update_layout(barmode='stack')
        
        
        return plot



