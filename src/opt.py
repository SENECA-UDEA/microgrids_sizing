# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:12 2022

@author: pmayaduque
"""

import pyomo.environ as pyo
from pyomo.core import value
from utilities import generation
import pandas as pd 
import time



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
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area for k in generators_dict.keys()})# generator area
    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area for k in batteries_dict.keys()})# generator area
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

    # TODO: Does this include the battery area?
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
      return model.p[k,t]<= generation(gen,t,forecast_df, model.Size[gen.id_gen]) * model.v[k,t]
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
      return  model.p[(k,t)] <= gen.c_max*model.v[k,t]
      #return  model.p[(k,t)] <= gen.c_max** model.v[k,t]*10 prueba para que active solar
    model.cmaxx_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cmaxx_rule)

    # Define Generación mínima
    def cminn_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] >= gen.c_min*model.v[k,t]
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

    # Define función objetivo
    def obj2_rule(model):
      return ((model.TNPC + model.TNPC_OP) * model.CRF) / sum( model.d[t] for t in model.HTIME)
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)
    #puse demanda para factibilidad

    return model

def make_model_operational(generators_dict=None, 
               forecast_df = None, 
               batteries_dict=None,  
               demand_df=None, 
               technologies_dict = None,  
               renovables_dict = None,
               nse = None, 
               TNPC = None,
               CRF = None,
               Size = None):
        
       
    model = pyo.ConcreteModel(name="Sizing microgrids Operational")
    
    # Sets
    model.GENERATORS = pyo.Set(initialize=[k for k in generators_dict.keys()])
    model.BATTERIES = pyo.Set(initialize=[l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize=[i for i in technologies_dict.keys()])
    model.RENOVABLES = pyo.Set(initialize=[r for r in renovables_dict.keys()])
    model.TECN_ALT = pyo.Set( initialize = [(i,j) for i in technologies_dict.keys() for j in technologies_dict[i]], ordered = False)
    model.HTIME = pyo.Set(initialize=[t for t in range(len(forecast_df))])

    # Parameters     
    model.d = pyo.Param(model.HTIME, initialize = demand_df) #demand     
    model.nse = pyo.Param(initialize=nse) # available unsupplied demand  
    model.TNPC = pyo.Param(initialize = TNPC)
    model.CRF = pyo.Param (initialize = CRF)
    model.Size = pyo.Param(model.GENERATORS, initialize = Size) #Size 
 

    # Variables
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
    model.TNPC_OP = pyo.Var(within=pyo.NonNegativeReals)
   
    # Restricciones


    # Restriccion de generación # SEPARAR FUNCIÓN GENERACIÓN 
    print("Start generation rule")
    def G_rule1 (model, k, t):
      gen = generators_dict[k]
      return model.p[k,t]<= generation(gen,t,forecast_df, model.Size[gen.id_gen]) * model.v[k,t]
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
      return  model.p[(k,t)] <= gen.c_max * model.v[k,t]
    #model.cmaxx_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cmaxx_rule)

    # Define Generación mínima
    def cminn_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] >= gen.c_min * model.v[k,t]
    #model.cminn_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cminn_rule)

    #Gestión de baterías
    def soc_rule(model, l, t):
      battery = batteries_dict[l]
      if t == 0:
          expr =  battery.eb_zero * (1-battery.alpha)
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
        return model.soc[l, t] <= battery.soc_max 
        #return model.soc[l, t] <= battery.soc_max * model.q[l] * 15  para que active la batería
    model.Bconstraint = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint_rule)

    #Mínimo nivel de SOC
    def Bconstraint2_rule(model, l, t):
      battery = batteries_dict[l]
      return model.soc[l, t] >= battery.soc_min 
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
   

    # Define restricción LPSP
    def lpspcons_rule(model, t):
      if model.d[t] > 0:
        return model.s_menos[t] / model.d[t]  <= model.nse 
      else:
        return pyo.Constraint.Skip
    model.lpspcons = pyo.Constraint(model.HTIME, rule=lpspcons_rule)
    
    
    #Función objetivo        

    # Define restricción TNPC OP
    def tnpcop_rule(model):
      
      return model.TNPC_OP ==  10*sum(model.s_menos[t] for t in model.HTIME) + sum(sum(generators_dict[k].va_op * model.p[k,t] for t in model.HTIME) for k in model.GENERATORS)
    model.tnpcop = pyo.Constraint(rule=tnpcop_rule)


    # Define función objetivo
    def obj2_rule(model):
      return ((model.TNPC + model.TNPC_OP) * model.CRF) / sum( model.d[t] for t in model.HTIME)
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)
    #puse demanda para factibilidad



    
    return model

def solve_model(model,
                optimizer='gurobi',
                mipgap=0.02,
                tee=True):
    solver = pyo.SolverFactory(optimizer)
    solver.options['MIPGap'] = mipgap
    timea = time.time()
    results = solver.solve(model, tee = tee)
    term_cond = results.solver.termination_condition
    term = {}
    # TODO: Check which other termination conditions may be interesting for us 
    # http://www.pyomo.org/blog/2015/1/8/accessing-solver
    if term_cond != pyo.TerminationCondition.optimal:
          term['Temination Condition'] = format(term_cond)
          execution_time = time.time() - timea
          term['Execution time'] = execution_time
          raise RuntimeError("Optimization failed.")

    else: 
          term['Temination Condition'] = format(term_cond)
          execution_time = time.time() - timea
          term['Execution time'] = execution_time    
    return results, term


class Results():
    def __init__(self, model):
        
        # general descriptives of tehe solution
        self.descriptive = {}
        
        # generators 
        generators = {}
        for k in model.GENERATORS:
           generators[k] = value(model.w[k])
        self.descriptive['generators'] = generators
        
        # technologies
        tecno_data = {}
        for i in model.TECHNOLOGIES:
           tecno_data[i] = value(model.y[i])
        self.descriptive['technologies'] = tecno_data
        
        bat_data = {}
        for l in model.BATTERIES:
           bat_data[l] = value(model.q[l])
        self.descriptive['batteries'] = bat_data
        
        com_data = {}
        for (i, j) in model.TECN_ALT:
            com_data[i, j] = value(model.x[i,j])  
        self.descriptive['comercial_alt'] = com_data
        
        area = 0
        for k in model.GENERATORS:
            area += value(model.w[k]) * model.gen_area[k]          
        for l in model.BATTERIES:
          area += value(model.q[l]) * model.bat_area[l]
        self.descriptive['area'] = area
            
            
        # objective function
        self.descriptive['LCOE'] = model.LCOE_value.expr()
        
        # Hourly data frame
        demand = pd.DataFrame(model.d.values(), columns=['demand'])
        
        generation = {k : [0]*len(model.HTIME) for k in model.GENERATORS}
        for (k,t), f in model.p.items():
          generation [k][t] = value(f)
        generation = pd.DataFrame(generation, columns=[*generation.keys()])
        
        # batery charge and discharge
        b_menos_data = {l+'_b-' : [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.b_menos.items():
          b_menos_data [l+'_b-'][t] = value(f)
        b_menos_df = pd.DataFrame(b_menos_data, columns=[*b_menos_data.keys()])

        b_mas_data = {l+'_b+': [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.b_mas.items():
          b_mas_data [l+'_b+'][t] = value(f)
        b_mas_df = pd.DataFrame(b_mas_data, columns=[*b_mas_data.keys()])
        
        soc_data = {l+'_soc' : [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.soc.items():
          soc_data [l+'_soc'][t] = value(f)
        soc_df = pd.DataFrame(soc_data, columns=[*soc_data.keys()])  
        
        # No supplied demand
        smenos_data = [0]*len(model.HTIME)
        lpsp_data = [0]*len(model.HTIME)
        for t in model.HTIME:
            smenos_data[t] = value(model.s_menos[t])
            if model.d[t] != 0:
              lpsp_data [t] = value(model.s_menos[t]) / value(model.d[t])
        
        smenos_df = pd.DataFrame(list(zip(smenos_data, lpsp_data)), columns = ['S-', 'LPSP'])
        

        self.df_results = pd.concat([demand, generation, b_menos_df, b_mas_df, soc_df, smenos_df ], axis=1) 
        


def create_results(model, 
                   demand_df,
                   generators_dict,
                   batteries_dict):
    
    gen_data = {}
    for k in model.GENERATORS:
       aux = []
       val = value(model.w[k])
       aux.append(val)
       gen_data[k] = aux
   
    gen_df = pd.DataFrame (gen_data.values(), index = [*gen_data.keys()], columns = ['w'])


    tecno_data = {}
    for i in model.TECHNOLOGIES:
       aux = []
       val = value(model.y[i])
       aux.append(val)
       tecno_data[i] = aux
   
    tecno_df = pd.DataFrame (tecno_data.values(), index = [*tecno_data.keys()], columns = ['y'])

    bat_data = {}
    for l in model.BATTERIES:
       aux = []
       val = value(model.q[l])
       aux.append(val)
       bat_data[l] = aux
   
    bat_df = pd.DataFrame (bat_data.values(), index = [*bat_data.keys()], columns = ['b'])
   

    p_data = {k : [0]*len(model.HTIME) for k in model.GENERATORS}
    for (k,t), f in model.p.items():
      p_data [k][t] = value(f)

    p_d = pd.DataFrame(p_data, columns=[*p_data.keys()])  

    smenos_data = [0]*len(model.HTIME)
    lpsp_data = [0]*len(model.HTIME)
    for t in model.HTIME:
        smenos_data[t] = value(model.s_menos[t])
        if model.d[t] != 0:
          lpsp_data [t] = value(model.s_menos[t]) / value(model.d[t])
    
    smenos_df = pd.DataFrame(smenos_data, columns = ['S-'])
    lpsp_df = pd.DataFrame(lpsp_data, columns = ['LPSP'])

    balance_df = pd.concat([p_d, demand_df, smenos_df,lpsp_df], axis=1)


    soc_data = {l : [0]*len(model.HTIME) for l in model.BATTERIES}
    for (l,t), f in model.soc.items():
      soc_data [l][t] = value(f)

    soc_df = pd.DataFrame(soc_data, columns=[*soc_data.keys()])

    b_menos_data = {l : [0]*len(model.HTIME) for l in model.BATTERIES}
    for (l,t), f in model.b_menos.items():
      b_menos_data [l][t] = value(f)

    b_menos_df = pd.DataFrame(b_menos_data, columns=[*b_menos_data.keys()])

    b_mas_data = {l : [0]*len(model.HTIME) for l in model.BATTERIES}
    for (l,t), f in model.b_mas.items():
      b_mas_data [l][t] = value(f)

    b_mas_df = pd.DataFrame(b_mas_data, columns=[*b_mas_data.keys()])

    obj_val = {}
    obj_val['LCOE'] = model.LCOE_value.expr()
 
    a_val = {}
    suma = 0
    for k in model.GENERATORS:
        ar =  value(model.w[k]) * generators_dict[k].area
        suma += ar  
    
    for l in model.BATTERIES:
      ar = value(model.q[l]) * batteries_dict[l].area
      suma += ar

    a_val['Area'] = suma

    com_data = {}

    for (i, j) in model.TECN_ALT:
            aux = []
            val = value(model.x[i,j])  
            aux.append(val)
            com_data[i, j] = aux
   
    com_df = pd.DataFrame (com_data.values(), index = [*com_data.keys()], columns = ['x'])
      
    return balance_df, soc_df, obj_val, gen_df, tecno_df, bat_df, com_df, a_val, b_menos_df, b_mas_df

