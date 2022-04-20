# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:12 2022

@author: pmayaduque
"""

import pyomo.environ as pyo



def make_model(generators_dict=None, 
               forecast_df = None, 
               batteries_dict=None,  
               demand_df=None, 
               technologies_dict = None,  
               renovables_dict = None,
               amax = None, 
               ir = None, 
               nse = None, 
               maxtec = None, 
               maxalt = None, 
               maxbat = None,
               years = None,
               Size = None):

    
    # Conjuntos
    model = pyo.ConcreteModel(name="Sizing microgrids")
    model.GENERATORS = pyo.Set(initialize=[k for k in generators_dict.keys()])
    model.BATTERIES = pyo.Set(initialize=[l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize=[i for i in technologies_dict.keys()])
    model.RENOVABLES = pyo.Set(initialize=[r for r in renovables_dict.keys()])
    model.TECN_ALT = pyo.Set( initialize = [(i,j) for i in technologies_dict.keys() for j in technologies_dict[i]], ordered = False)
    model.HTIME = pyo.Set(initialize=[t for t in range(len(forecast_df))])

    # Parámetros    
    model.amax = pyo.Param(initialize=amax) #Área máxima    
    model.d = pyo.Param(model.HTIME, initialize = demand_df) #demanda    
    model.ir = pyo.Param(initialize=ir) #tasa de interés    
    model.nse = pyo.Param(initialize=nse) #demanda no abastecida permitida    
    model.maxtec = pyo.Param(initialize = maxtec) #máximo de tecnologías    
    model.maxalt = pyo.Param(initialize = maxalt) #máximo de alternativas    
    model.maxbat = pyo.Param(initialize = maxbat) #máximo de baterías
    model.t_years = pyo.Param(initialize = years) # Número de años para CRF   
    CRF_calc = (model.ir * (1 + model.ir)**(model.t_years))/((1 + model.ir)**(model.t_years)-1) #CRF para LCOE
    model.CRF = pyo.Param(initialize = CRF_calc)
    model.Size = pyo.Param(model.GENERATORS, initialize = Size) #Size   


    # Variables
    model.y = pyo.Var(model.TECHNOLOGIES, within=pyo.Binary)
    model.x = pyo.Var(model.TECN_ALT, within=pyo.Binary)
    model.w = pyo.Var(model.GENERATORS, within=pyo.Binary)
    model.q = pyo.Var(model.BATTERIES, within=pyo.Binary)
    model.v = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.Binary)
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.b_mas = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.b_menos = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.p_g = pyo.Var(model.TECHNOLOGIES, model.HTIME, within=pyo.NonNegativeReals)
    model.p_c = pyo.Var(model.TECN_ALT, model.HTIME, within=pyo.NonNegativeReals)
    model.p_ren = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.s_menos = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.p_tot = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.bd = pyo.Var(model.HTIME, within=pyo.Binary)
    model.bc = pyo.Var(model.HTIME, within=pyo.Binary)
    model.TNPC = pyo.Var(within=pyo.NonNegativeReals)
    model.TNPC_OP = pyo.Var(within=pyo.NonNegativeReals)
   
    # Restricciones

    #define regla de tecnologías y alternativas comerciales
    def xy_rule (model,i,j):
        return model.x[i,j] <= model.y[i]
    model.xy_rule = pyo.Constraint(model.TECN_ALT, rule = xy_rule)
    
    #define regla de alternativas comerciales y generadores
    def wk_rule(model, i, j, k):
        gen = generators_dict[k]
        if gen.tec == i and gen.alt == j:
            return model.w[k] <= model.x[i,j] 
        else:
            return pyo.Constraint.Skip
    model.wk_rule = pyo.Constraint(model.TECN_ALT, model.GENERATORS, rule=wk_rule)

    #define regla de alternativas comerciales y baterías
    def ql_rule(model, i, j, l):
        bat = batteries_dict[l]
        if  (bat.tec == i) and (bat.alt == j):
            return model.q[l] <= model.x[i,j] 
        else:
            return pyo.Constraint.Skip
    model.ql_rule = pyo.Constraint(model.TECN_ALT, model.BATTERIES, rule=ql_rule)

    # Define restricción área
    def area_rule(model):
      return  sum(generators_dict[k].area*model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].area*model.q[l] for l in model.BATTERIES) <= model.amax
    model.area_rule = pyo.Constraint(rule=area_rule)

    # Define regla de activar o desactivar generadores por cada periodo de tiempo
    def vkt_rule(model,k,t):
        return model.v[k,t] <= model.w[k]
    model.vkt_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=vkt_rule)

    # Restriccion de generación # SEPARAR FUNCIÓN GENERACIÓN 
    print("Start generation rule")
    def G_rule1 (model, k, t):
      gen = generators_dict[k]
      return model.p[k,t]<= generacion(gen,t,forecast_df, model.Size[gen.id_gen]) * model.v[k,t]
    model.G_rule1 = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_rule1)
    print("End generation rule")
    # Define Balance de energía
    def balance_rule(model, t):
      return  sum(model.p[(k,t)] for k in model.GENERATORS) +  sum(model.b_menos[(l,t)] for l in model.BATTERIES) + model.s_menos[t] == model.d[t]  + sum(model.b_mas[(l,t)] for l in model.BATTERIES)
    model.balance_rule = pyo.Constraint(model.HTIME, rule=balance_rule)

    # Define  energía total
    def ptot_rule(model,t):
      return sum( model.p[(k,t)] for k in model.GENERATORS) == model.p_tot[t]
    model.ptot_rule = pyo.Constraint(model.HTIME, rule=ptot_rule)
    
    # Define energía por tecnología
    def ptec_rule(model,i,t): 
      if i != 'B': 
          return sum( model.p[(k,t)] for k in model.GENERATORS if generators_dict[k].tec == i) == model.p_g[i,t]   
      else:
          return sum(model.b_menos[(l,t)] for l in model.BATTERIES) == model.p_g[i,t]
    model.ptec_rule = pyo.Constraint(model.TECHNOLOGIES, model.HTIME, rule=ptec_rule)


    # Define energía por alternativa 
    def palt_rule(model,i,j,t): 
      if i != 'B': 
          return sum(model.p[(k,t)] for k in model.GENERATORS if generators_dict[k].tec == i and generators_dict[k].alt == j) == model.p_c[i,j,t]
      else:
          return sum(model.b_menos[(l,t)] for l in model.BATTERIES if  batteries_dict[l].tec == i and batteries_dict[l].alt == j) == model.p_c[i,j,t]
    model.palt_rule = pyo.Constraint(model.TECN_ALT, model.HTIME, rule=palt_rule)

    # Define energía RENOVABLE 
    def pren_rule(model,t):
      return sum( model.p_g[(r,t)] for r in model.RENOVABLES) == model.p_ren[t]
    model.pren_rule = pyo.Constraint( model.HTIME, rule=pren_rule)
    
    # Define Capacidad máxima
    def cmaxx_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] <= gen.c_max** model.v[k,t]
      #return  model.p[(k,t)] <= gen.c_max** model.v[k,t]*10 prueba para que active solar
    model.cmaxx_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cmaxx_rule)

    # Define Generación mínima
    def cminn_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] >= gen.c_min** model.v[k,t]
    model.cminn_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cminn_rule)

    #Gestión de baterías
    def soc_rule(model, l, t):
      battery = batteries_dict[l]
      if t == 0:
          expr =  model.q[l] * battery.eb_zero * (1-battery.alpha)
          expr += model.b_mas[l,t] * battery.efc
          expr -= (model.b_menos[l, t]/battery.efd)
          return model.soc[l, t] == expr 
      else:
          expr = model.soc[l, t-1] * (1-battery.alpha)
          expr += model.b_mas[l, t] * battery.efc
          expr -= (model.b_menos[l, t]/battery.efd)
          return model.soc[l, t] == expr
    model.soc_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule=soc_rule)

    #Máximo nivel de SOC
    def Bconstraint_rule(model, l, t):
        battery = batteries_dict[l]
        return model.soc[l, t] <= battery.soc_max * model.q[l]
        #return model.soc[l, t] <= battery.soc_max * model.q[l] * 15  para que active la batería
    model.Bconstraint = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint_rule)

    #Mínimo nivel de SOC
    def Bconstraint2_rule(model, l, t):
      battery = batteries_dict[l]
      return model.soc[l, t] >= battery.soc_min * model.q[l]
    model.Bconstraint2 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint2_rule)

    #Mínimo nivel que puede entrar a la batería
    def Bconstraint3_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_mas[l, t] >= battery.soc_min * model.bc[t]
    model.Bconstraint3 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint3_rule)

    #mínimo nivel que puede salir de la batería
    def Bconstraint4_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_menos[l, t] >= battery.soc_min * model.bd[t]
    model.Bconstraint4 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint4_rule)

    #máximo nivel que puede entrar a la batería
    def Bconstraint5_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_mas[l, t] <= battery.soc_max * model.bc[t] 
    model.Bconstraint5 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint5_rule)

    #máximo nivel que puede salir a la batería
    def Bconstraint6_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_menos[l, t] <= battery.soc_max * model.bd[t] 
    model.Bconstraint6 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint6_rule)
    
    # Define control carga y descarga 
    def bcbd_rule(model,t):
      return  model.bc[t] + model.bd[t] <= 1
    model.bcbd_rule = pyo.Constraint(model.HTIME, rule=bcbd_rule)
   
    # Define restricción max_tec
    def maxtec_rule(model):
      return sum(model.y[i] for i in model.TECHNOLOGIES) <= model.maxtec
    #model.maxtec_rule = pyo.Constraint(rule=maxtec_rule)

    #Define restricción max_alt
    def maxalt_rule (model, i):
      return sum (model.x[i,j] for  j in technologies_dict[i]) <= model.maxalt
    #model.maxalt_rule = pyo.Constraint(model.TECHNOLOGIES, rule = maxalt_rule)
    #Define restricción max_bat
    def maxbat_rule(model):
        return sum(model.q[l] for l in model.BATTERIES)  <= model.maxbat
    #model.maxbat_rule = pyo.Constraint(rule=maxbat_rule)


    #Función objetivo
        
    # Define restricción TNPC
    def tnpcc_rule(model):
            expr = 10*sum(model.s_menos[t] for t in model.HTIME)
            expr += sum(generators_dict[k].cost_up *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_up * model.q[l] for l in model.BATTERIES) 
            expr += sum(generators_dict[k].cost_r *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_r * model.q[l]  for l in model.BATTERIES) 
            expr += sum(generators_dict[k].cost_om *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_om * model.q[l]  for l in model.BATTERIES)
            expr -= sum(generators_dict[k].cost_s *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_s * model.q[l]  for l in model.BATTERIES)
            return model.TNPC == expr
    model.tnpcc = pyo.Constraint(rule=tnpcc_rule)

    # Define restricción TNPC OP
    def tnpcop_rule(model):
      return model.TNPC_OP ==  sum(sum(generators_dict[k].va_op * model.p[k,t] for t in model.HTIME) for k in model.GENERATORS)
    model.tnpcop = pyo.Constraint(rule=tnpcop_rule)

    # Define restricción LPSP
    def lpspcons_rule(model, t):
      if model.d[t] > 0:
        return model.s_menos[t] / model.d[t]  <= model.nse 
      else:
        return pyo.Constraint.Skip
    model.lpspcons = pyo.Constraint(model.HTIME, rule=lpspcons_rule)



    '''
    LPSP
    def obj1_rule(model):
        return sum(model.s_menos[t] / model.d[t] for t in model.HTIME)
    model.LPSP_value = pyo.Objective(sense=minimize, rule=obj1_rule)

    #Ambiental restricción
    def amb_rule(model, t):
      return model.p_ren[t] / model.d[t]  >= model.renfrac 
    model.amb_rule = pyo.Constraint(model.HTIME, rule=amb_rule)
    '''

    # Define función objetivo2
    def obj2_rule(model):
      return ((model.TNPC + model.TNPC_OP) * model.CRF) / sum( model.d[t] for t in model.HTIME)
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)
    #puse demanda para factibilidad

    return model