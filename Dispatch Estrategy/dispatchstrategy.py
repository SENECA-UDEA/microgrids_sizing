# -*- coding: utf-8 -*-

def strategy (batteries_dict, generators_dict):
    
    d=0
    s=0
    b=0
    w=0    
    
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

def d (generators_dict, batteries_dict, demand_df, instance_data, cost_data, CRF, fuel_cost):
    
    auxiliar_dict_generator = []
    lcoe_op = 0
    lcoe_inf = 0
    total_gen = 0
    for g in generators_dict.values():
        auxiliar_dict_generator[g.id_gen] = g.DG_max
        lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * CRF + g.cost_fopm
    
    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=True) 
    p = [[]]
    cost =[[]]
    costsplus = []
    for t in demand_df:
        demand_tobe_covered = demand_df[t]
        for i in sorted_generators:
             gen = generators_dict[i]
             if (demand_tobe_covered == 0):
                 p[i,t] == 0
                 cost[i,t]==0
                 cont = 0
             elif (gen.DG_max >= demand_tobe_covered):
                 cont = 1
                 p[i,t] = max(demand_tobe_covered,gen.DG_min)
                 demand_tobe_covered = 0
                 if (p[i,t] > demand_tobe_covered):
                     costsplus[t] = (p[i,t] - demand_tobe_covered) * instance_data["cost_sminus"]
             else:
                cont = 1
                p[i,t] = gen.DG_max
                demand_tobe_covered = demand_tobe_covered - gen.DG_max
   
             cost[i,t] = (gen.f0 * gen.DG_max * cont + gen.f1 * p[i,t]) * fuel_cost
             lcoe_op += cost[i,t]  
             total_gen += p[i,t]
                
    lcoe = (lcoe_inf + lcoe_op)/total_gen
    df = cost + costsplus + p
    return lcoe, df
    
    

def ds (generators_dict, batteries_dict, demand_df, instance_data, cost_data, CRF, fuel_cost, delta):
    
    auxiliar_dict_generator = []
    dict_ren = []
    demand_tobe_covered = []
    lcoe_op = 0
    lcoe_inf = 0
    total_gen = 0
    costsplus = []
    demand_tobe_covered
    for g in generators_dict.values():
        if (g.tec == 'D'):
            auxiliar_dict_generator[g.id_gen] = g.DG_max
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * CRF + g.cost_fopm
        else:
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
            dict_ren.append(g.id_gen)
    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=True) 
    ref = generators_dict[sorted_generators[len(sorted_generators)]].DG_min
    for t in demand_df['t']:
        generation_ren = sum(generators_dict[i].gen_rule[t] for i in dict_ren)
        if generation_ren > (demand_df - ref):
            costsplus[t] = generation_ren + ref - demand_df
            demand_tobe_covered = 0
        else:
            demand_tobe_covered = demand_df['t'] - generation_ren
        
    p = [[]]
    cost =[[]]
    
    for t in demand_df['t']:
        for i in sorted_generators:
             gen = generators_dict[i]
             if (demand_tobe_covered == 0):
                 p[i,t] == 0
                 cost[i,t]==0
                 cont = 0
             elif (gen.DG_max >= demand_tobe_covered):
                 cont = 1
                 p[i,t] = max(demand_tobe_covered,gen.DG_min)
                 demand_tobe_covered = 0
                 if (p[i,t] > demand_tobe_covered):
                     costsplus[t] = (p[i,t] - demand_tobe_covered) * instance_data["cost_sminus"]
             else:
                cont = 1
                p[i,t] = gen.DG_max
                demand_tobe_covered = demand_tobe_covered - gen.DG_max
   
             cost[i,t] = (gen.f0 * gen.DG_max * cont + gen.f1 * p[i,t]) * fuel_cost
             lcoe_op += cost[i,t]  
             total_gen += p[i,t]
                
    lcoe = (lcoe_inf + lcoe_op)/total_gen
    df = cost + costsplus + p
    return lcoe, df    

def dw (generators_dict, batteries_dict, demand_df, instance_data, cost_data, CRF, fuel_cost, delta):
    
    auxiliar_dict_generator = []
    dict_ren = []
    demand_tobe_covered = []
    lcoe_op = 0
    lcoe_inf = 0
    total_gen = 0
    costsplus = []
    demand_tobe_covered
    for g in generators_dict.values():
        if (g.tec == 'D'):
            auxiliar_dict_generator[g.id_gen] = g.DG_max
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * CRF + g.cost_fopm
        else:
            lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
            dict_ren.append(g.id_gen)
    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=True) 
    ref = generators_dict[sorted_generators[len(sorted_generators)]].DG_min
    for t in demand_df['t']:
        generation_ren = sum(generators_dict[i].gen_rule[t] for i in dict_ren)
        if generation_ren > (demand_df - ref):
            costsplus[t] = generation_ren + ref - demand_df
            demand_tobe_covered = 0
        else:
            demand_tobe_covered = demand_df['t'] - generation_ren
        
    p = [[]]
    cost =[[]]
    
    for t in demand_df['t']:
        for i in sorted_generators:
             gen = generators_dict[i]
             if (demand_tobe_covered == 0):
                 p[i,t] == 0
                 cost[i,t]==0
                 cont = 0
             elif (gen.DG_max >= demand_tobe_covered):
                 cont = 1
                 p[i,t] = max(demand_tobe_covered,gen.DG_min)
                 demand_tobe_covered = 0
                 if (p[i,t] > demand_tobe_covered):
                     costsplus[t] = (p[i,t] - demand_tobe_covered) * instance_data["cost_sminus"]
             else:
                cont = 1
                p[i,t] = gen.DG_max
                demand_tobe_covered = demand_tobe_covered - gen.DG_max
   
             cost[i,t] = (gen.f0 * gen.DG_max * cont + gen.f1 * p[i,t]) * fuel_cost
             lcoe_op += cost[i,t]  
             total_gen += p[i,t]
                
    lcoe = (lcoe_inf + lcoe_op)/total_gen
    df = cost + costsplus + p
    return lcoe, df   


def bs (generators_dict, batteries_dict, demand_df, instance_data, cost_data, CRF, fuel_cost, delta):
    
    auxiliar_dict_generator = []
    dict_ren = []
    demand_tobe_covered = []
    lcoe_op = 0
    lcoe_inf = 0
    total_gen = 0
    costsplus = []
    demand_tobe_covered
    for g in generators_dict.values():
        lcoe_inf = (g.cost_up + g.cost_r - g.cost_s) * delta * CRF + g.cost_fopm
        dict_ren.append(g.id_gen)
    
    for b in batteries_dict.values():
        auxiliar_dict_generator[b.id_bat] = b.SOC_max

    sorted_generators = sorted(auxiliar_dict_generator, key=auxiliar_dict_generator.get,reverse=True) 

    for t in demand_df['t']:
        generation_ren = sum(generators_dict[i].gen_rule[t] for i in dict_ren)
        if generation_ren > (demand_df):
            for bat in sorted_generators:
                b = batteries_dict[bat]
                surplus = generation_ren - demand_df
                generation_ren = surplus
                if (surplus > 0):
                    b.soc[t]= min(b.SOC_MAX - surplus, surplus)
            costsplus[t] = generation_ren - demand_df
            demand_tobe_covered = 0
        else:
            demand_tobe_covered = demand_df['t'] - generation_ren
        
    pb = [[]]
    cost =[[]]
    costsminus = [[]]
    sminus = []
    for t in demand_df['t']:
        for i in sorted_generators:
             gen = generators_dict[i]
             if (demand_tobe_covered == 0):
                 pb[i,t] == 0
                 cost[i,t]==0
                 cont = 0
             else:
                 cont = 1
                 pb[i,t] = min(demand_tobe_covered,gen.SOC_max - gen.Soc[t])
                 if (pb[i,t] < demand_tobe_covered):
                     sminus[t] = demand_tobe_covered
                     costsminus[t] = (pb[i,t] - demand_tobe_covered) * instance_data["cost_sminus"]

             demand_tobe_covered = demand_tobe_covered - gen.DG_max
 
             total_gen += pb[i,t]
                
    lcoe = (lcoe_inf + lcoe_op)/total_gen
    df = cost + costsplus + pb
    a = sum(sminus/d)
    if a >= 1.5:
        state = "infeasible"
    return lcoe, df, state