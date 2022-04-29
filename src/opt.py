# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:12 2022

@author: pmayaduque
"""

import pyomo.environ as pyo
from pyomo.core import value
from utilities import generation
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
import pandas as pd 
import time



def make_model(generators_dict=None, 
               forecast_df = None, 
               batteries_dict=None,  
               demand_df=None, 
               technologies_dict = None,  
               renewables_dict = None,
               amax = None, 
               ir = None, 
               nse = None, 
               maxtec = None, 
               maxbr = None,
               years = None,
               tlpsp = None):

    
    # Sets
    model = pyo.ConcreteModel(name="Sizing microgrids")
    model.GENERATORS = pyo.Set(initialize=[k for k in generators_dict.keys()])
    model.BATTERIES = pyo.Set(initialize=[l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize=[i for i in technologies_dict.keys()])
    model.RENEWABLES = pyo.Set(initialize=[r for r in renewables_dict.keys()])
    model.TEC_BRAND = pyo.Set( initialize = [(i,j) for i in technologies_dict.keys() for j in technologies_dict[i]], ordered = False)
    model.HTIME = pyo.Set(initialize=[t for t in range(len(forecast_df))])

    # Parameters
    model.amax = pyo.Param(initialize=amax) #Maximum area
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area for k in generators_dict.keys()})# Generator area
    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area for k in batteries_dict.keys()})# Battery area
    model.d = pyo.Param(model.HTIME, initialize = demand_df) #demand    
    model.ir = pyo.Param(initialize=ir) #Interest rate
    model.nse = pyo.Param(initialize=nse) #Available not supplied demand  
    model.maxtec = pyo.Param(initialize = maxtec) #Maximum technologies  
    model.maxbr = pyo.Param(model.TECHNOLOGIES, initialize = maxbr) #Maximum brand by each technology  
    model.t_years = pyo.Param(initialize = years) # Number of years for the project, for CRF
    CRF_calc = (model.ir * (1 + model.ir)**(model.t_years))/((1 + model.ir)**(model.t_years)-1) #CRF to LCOE
    model.CRF = pyo.Param(initialize = CRF_calc)  
    model.tlpsp = pyo.Param (initialize = tlpsp) #LPSP Time for moving average


    # Variables
    model.y = pyo.Var(model.TECHNOLOGIES, within=pyo.Binary)
    model.x = pyo.Var(model.TEC_BRAND, within=pyo.Binary)
    model.w = pyo.Var(model.GENERATORS, within=pyo.Binary)
    model.q = pyo.Var(model.BATTERIES, within=pyo.Binary)
    model.v = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.Binary)
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.b_charge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.b_discharge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.p_g = pyo.Var(model.TECHNOLOGIES, model.HTIME, within=pyo.NonNegativeReals)
    model.p_c = pyo.Var(model.TEC_BRAND, model.HTIME, within=pyo.NonNegativeReals)
    model.p_ren = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.s_minus = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.p_tot = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.bd = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary)
    model.bc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary)
    model.TNPC = pyo.Var(within=pyo.NonNegativeReals)
    model.TNPC_OP = pyo.Var(within=pyo.NonNegativeReals)
   
    # Constraints

    #Defines rule of technologies and brands
    def xy_rule (model,i,j):
        return model.x[i,j] <= model.y[i]
    model.xy_rule = pyo.Constraint(model.TEC_BRAND, rule = xy_rule)
    
    #Defines rule of brands and generators
    def wk_rule(model, i, j, k):
        gen = generators_dict[k]
        if gen.tec == i and gen.br == j:
            return model.w[k] <= model.x[i,j] 
        else:
            return pyo.Constraint.Skip
    model.wk_rule = pyo.Constraint(model.TEC_BRAND, model.GENERATORS, rule=wk_rule)

    #Defines rule of brands and batteries
    def ql_rule(model, i, j, l):
        bat = batteries_dict[l]
        if  (bat.tec == i) and (bat.br == j):
            return model.q[l] <= model.x[i,j] 
        else:
            return pyo.Constraint.Skip
    model.ql_rule = pyo.Constraint(model.TEC_BRAND, model.BATTERIES, rule=ql_rule)

    # Defines area rule
    def area_rule(model):
      return  sum(model.gen_area[k]*model.w[k] for k in model.GENERATORS) + sum(model.bat_area[l]*model.q[l] for l in model.BATTERIES) <= model.amax
    model.area_rule = pyo.Constraint(rule=area_rule)

    # Defines rule to activate or deactivate generators for each period of time
    def vkt_rule(model,k,t):
        return model.v[k,t] <= model.w[k]
    model.vkt_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=vkt_rule)

    # Generation rule
    print("Start generation rule")
    def G_rule1 (model, k, t):
      gen = generators_dict[k]
      return model.p[k,t]<= generation(gen,t,forecast_df) * model.v[k,t]
    model.G_rule1 = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_rule1)
    print("End generation rule")
    
    # Defines balance rule
    def balance_rule(model, t):
      return  sum(model.p[(k,t)] for k in model.GENERATORS) +  sum(model.b_discharge[(l,t)] for l in model.BATTERIES) + model.s_minus[t] == model.d[t]  + sum(model.b_charge[(l,t)] for l in model.BATTERIES)
    model.balance_rule = pyo.Constraint(model.HTIME, rule=balance_rule)

    # Defines total energy generated
    def ptot_rule(model,t):
      return sum( model.p[(k,t)] for k in model.GENERATORS) == model.p_tot[t]
    model.ptot_rule = pyo.Constraint(model.HTIME, rule=ptot_rule)
    
    # Defines energy generated by each technology
    def ptec_rule(model,i,t): 
      if i != 'B': 
          return sum( model.p[(k,t)] for k in model.GENERATORS if generators_dict[k].tec == i) == model.p_g[i,t]   
      else:
          return sum(model.b_discharge[(l,t)] for l in model.BATTERIES) == model.p_g[i,t]
    model.ptec_rule = pyo.Constraint(model.TECHNOLOGIES, model.HTIME, rule=ptec_rule)


    # Defines energy generated by each brand
    def pbrand_rule(model,i,j,t): 
      if i != 'B': 
          return sum(model.p[(k,t)] for k in model.GENERATORS if generators_dict[k].tec == i and generators_dict[k].br == j) == model.p_c[i,j,t]
      else:
          return sum(model.b_discharge[(l,t)] for l in model.BATTERIES if  batteries_dict[l].tec == i and batteries_dict[l].br == j) == model.p_c[i,j,t]
    model.pbrand_rule = pyo.Constraint(model.TEC_BRAND, model.HTIME, rule=pbrand_rule)

    # Defines renewable energy
    def pren_rule(model,t):
      return sum( model.p_g[(r,t)] for r in model.RENEWABLES) == model.p_ren[t]
    model.pren_rule = pyo.Constraint( model.HTIME, rule=pren_rule)
    
    # Defines maximum capacity
    def cmaxx_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] <= gen.c_max*model.v[k,t]
    model.cmaxx_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cmaxx_rule)

    # Defines minimum power to activate the generator
    def cminn_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] >= gen.c_min*model.v[k,t]
    model.cminn_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cminn_rule)

    #Batteries management
    def soc_rule(model, l, t):
      battery = batteries_dict[l]
      if t == 0:
          expr =  model.q[l] * battery.eb_zero * (1-battery.alpha)
          expr += model.b_charge[l,t] * battery.efc
          expr -= (model.b_discharge[l, t]/battery.efd)
          return model.soc[l, t] == expr 
      else:
          expr = model.soc[l, t-1] * (1-battery.alpha)
          expr += model.b_charge[l, t] * battery.efc
          expr -= (model.b_discharge[l, t]/battery.efd)
          return model.soc[l, t] == expr
    model.soc_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule=soc_rule)

    #Highest level of SOC
    def Bconstraint_rule(model, l, t):
        battery = batteries_dict[l]
        return model.soc[l, t] <= battery.soc_max * model.q[l]
    model.Bconstraint = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint_rule)

    #Minimun SOC level
    def Bconstraint2_rule(model, l, t):
      battery = batteries_dict[l]
      return model.soc[l, t] >= battery.soc_min * model.q[l]
    model.Bconstraint2 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint2_rule)

    #Minimum level of energy that can enter to the battery
    def Bconstraint3_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_charge[l, t] >= battery.soc_min * model.bc[l, t]
    model.Bconstraint3 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint3_rule)

    #Minimum level of energy that the battery can give to the microgrid
    def Bconstraint4_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_discharge[l, t] >= battery.soc_min * model.bd[l, t]
    model.Bconstraint4 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint4_rule)

    
    #Maximum level of energy that can enter to the battery
    def Bconstraint5_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_charge[l, t] <= battery.soc_max * model.bc[l, t] 
    model.Bconstraint5 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint5_rule)

    #Maximum level of energy that the battery can give to the microgrid
    def Bconstraint6_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_discharge[l, t] <= battery.soc_max * model.bd[l, t] 
    model.Bconstraint6 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint6_rule)
    
    # Charge and discharge control 
    def bcbd_rule(model, l, t):
      return  model.bc[l, t] + model.bd[l, t] <= 1
    model.bcbd_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule=bcbd_rule)
   
    # Defines constraint of maximum number of technologies
    def maxtec_rule(model):
      return sum(model.y[i] for i in model.TECHNOLOGIES) <= model.maxtec
    model.maxtec_rule = pyo.Constraint(rule=maxtec_rule)

    # Defines constraint of maximum number of brands
    def maxbr_rule (model, i):
        return sum (model.x[i,j] for  j in technologies_dict[i]) <= model.maxbr[i]
    model.maxbr_rule = pyo.Constraint(model.TECHNOLOGIES, rule = maxbr_rule)

    #Objective function
        
    # Defines TNPC constraint
    def tnpcc_rule(model): 
            expr = sum(generators_dict[k].cost_up *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_up * model.q[l] for l in model.BATTERIES) 
            expr += sum(generators_dict[k].cost_r *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_r * model.q[l]  for l in model.BATTERIES) 
            expr += sum(generators_dict[k].cost_om *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_om * model.q[l]  for l in model.BATTERIES)
            expr -= sum(generators_dict[k].cost_s *model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_s * model.q[l]  for l in model.BATTERIES)
            return model.TNPC == expr
    model.tnpcc = pyo.Constraint(rule=tnpcc_rule)

    # Define TNPC operative constraint
    def tnpcop_rule(model):
        #TODO check cost unsupplied
        expr2 = 10*sum(model.s_minus[t] for t in model.HTIME)
        expr2 += sum(sum(generators_dict[k].va_op * model.p[k,t] for t in model.HTIME) for k in model.GENERATORS)
        return model.TNPC_OP == expr2
    model.tnpcop = pyo.Constraint(rule=tnpcop_rule)
    
    
    # Defines LPSP constraint
    def lpspcons_rule(model, t):
      if t >= (model.tlpsp - 1):
        rev = sum(model.d[t] for t in range((t-model.tlpsp+1), t+1)) 
        if rev > 0:
          return sum(model.s_minus[t] for t in range((t-model.tlpsp+1), t+1)) / rev  <= model.nse 
        else:
          return pyo.Constraint.Skip
      else:
        return pyo.Constraint.Skip
    model.lpspcons = pyo.Constraint(model.HTIME, rule=lpspcons_rule)


    '''
    LPSP
    def obj1_rule(model):
        return sum(model.s_minus[t] / model.d[t] for t in model.HTIME)
    model.LPSP_value = pyo.Objective(sense=minimize, rule=obj1_rule)

    #Enviromental constraint
    def amb_rule(model, t):
      return model.p_ren[t] / model.d[t]  >= model.renfrac 
    model.amb_rule = pyo.Constraint(model.HTIME, rule=amb_rule)
    '''

    # Defines objective function
    def obj2_rule(model):
      return ((model.TNPC + model.TNPC_OP) * model.CRF) / sum( model.d[t] for t in model.HTIME)
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)
    #I put demand for linearity

    return model





def make_model_operational(generators_dict=None, 
               forecast_df = None, 
               batteries_dict=None,  
               demand_df=None, 
               technologies_dict = None,  
               renewables_dict = None,               
               amax = None, 
               nse = None, 
               TNPC = None,
               CRF = None,
               tlpsp = None):
        
       
    model = pyo.ConcreteModel(name="Sizing microgrids Operational")
    
    # Sets
    model.GENERATORS = pyo.Set(initialize=[k for k in generators_dict.keys()])
    model.BATTERIES = pyo.Set(initialize=[l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize=[i for i in technologies_dict.keys()])
    model.RENEWABLES = pyo.Set(initialize=[r for r in renewables_dict.keys()])
    model.TEC_BRAND = pyo.Set( initialize = [(i,j) for i in technologies_dict.keys() for j in technologies_dict[i]], ordered = False)
    model.HTIME = pyo.Set(initialize=[t for t in range(len(forecast_df))])

    # Parameters 
    model.amax = pyo.Param(initialize=amax) #Maximum area
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area for k in generators_dict.keys()})# Generator area
    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area for k in batteries_dict.keys()})# Battery area    
    model.d = pyo.Param(model.HTIME, initialize = demand_df) #demand     
    model.nse = pyo.Param(initialize=nse) # Available unsupplied demand  
    model.TNPC = pyo.Param(initialize = TNPC)
    model.CRF = pyo.Param (initialize = CRF)
    model.tlpsp = pyo.Param (initialize = tlpsp)

    # Variables
    model.v = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.Binary)
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.b_charge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.b_discharge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals)
    model.p_g = pyo.Var(model.TECHNOLOGIES, model.HTIME, within=pyo.NonNegativeReals)
    model.p_c = pyo.Var(model.TEC_BRAND, model.HTIME, within=pyo.NonNegativeReals)
    model.p_ren = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.s_minus = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.p_tot = pyo.Var(model.HTIME, within=pyo.NonNegativeReals)
    model.bd = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary)
    model.bc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary)
    model.TNPC_OP = pyo.Var(within=pyo.NonNegativeReals)
   
    # Constraints


    # Generation rule
    print("Start generation rule")
    def G_rule1 (model, k, t):
      gen = generators_dict[k]
      return model.p[k,t]<= generation(gen,t,forecast_df) * model.v[k,t]
    model.G_rule1 = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_rule1)
    print("End generation rule")

    # Defines energy balance
    def balance_rule(model, t):
      return  sum(model.p[(k,t)] for k in model.GENERATORS) +  sum(model.b_discharge[(l,t)] for l in model.BATTERIES) + model.s_minus[t] == model.d[t]  + sum(model.b_charge[(l,t)] for l in model.BATTERIES)
    model.balance_rule = pyo.Constraint(model.HTIME, rule=balance_rule)

    # Defines total energy generated by each period time
    def ptot_rule(model,t):
      return sum( model.p[(k,t)] for k in model.GENERATORS) == model.p_tot[t]
    model.ptot_rule = pyo.Constraint(model.HTIME, rule=ptot_rule)
    
    # Defines energy generated by each technology
    def ptec_rule(model,i,t): 
      if i != 'B': 
          return sum( model.p[(k,t)] for k in model.GENERATORS if generators_dict[k].tec == i) == model.p_g[i,t]   
      else:
          return sum(model.b_discharge[(l,t)] for l in model.BATTERIES) == model.p_g[i,t]
    model.ptec_rule = pyo.Constraint(model.TECHNOLOGIES, model.HTIME, rule=ptec_rule)


    # Defines energy generated by each brand
    def pbrand_rule(model,i,j,t): 
      if i != 'B': 
          return sum(model.p[(k,t)] for k in model.GENERATORS if generators_dict[k].tec == i and generators_dict[k].br == j) == model.p_c[i,j,t]
      else:
          return sum(model.b_discharge[(l,t)] for l in model.BATTERIES if  batteries_dict[l].tec == i and batteries_dict[l].br == j) == model.p_c[i,j,t]
    model.pbrand_rule = pyo.Constraint(model.TEC_BRAND, model.HTIME, rule=pbrand_rule)

    # Defines renewable energy generated
    def pren_rule(model,t):
      return sum( model.p_g[(r,t)] for r in model.RENEWABLES) == model.p_ren[t]
    model.pren_rule = pyo.Constraint( model.HTIME, rule=pren_rule)
    
    # Defines maximum capacity
    def cmaxx_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] <= gen.c_max * model.v[k,t]
    model.cmaxx_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cmaxx_rule)

    # Define minimum power to activate the generator
    def cminn_rule(model,k, t):
      gen = generators_dict[k]
      return  model.p[(k,t)] >= gen.c_min * model.v[k,t]
    model.cminn_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cminn_rule)

    # Batteries management
    def soc_rule(model, l, t):
      battery = batteries_dict[l]
      if t == 0:
          expr =  battery.eb_zero * (1-battery.alpha)
          expr += model.b_charge[l,t] * battery.efc
          expr -= (model.b_discharge[l, t]/battery.efd)
          return model.soc[l, t] == expr 
      else:
          expr = model.soc[l, t-1] * (1-battery.alpha)
          expr += model.b_charge[l, t] * battery.efc
          expr -= (model.b_discharge[l, t]/battery.efd)
          return model.soc[l, t] == expr
    model.soc_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule=soc_rule)

    # Highest SOC level
    def Bconstraint_rule(model, l, t):
        battery = batteries_dict[l]
        return model.soc[l, t] <= battery.soc_max 
    model.Bconstraint = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint_rule)

    # Minimum SOC level
    def Bconstraint2_rule(model, l, t):
      battery = batteries_dict[l]
      return model.soc[l, t] >= battery.soc_min 
    model.Bconstraint2 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint2_rule)
    
    # Minimum level of energy that can enter to the battery
    def Bconstraint3_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_charge[l, t] >= battery.soc_min * model.bc[l, t]
    model.Bconstraint3 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint3_rule)

    # Minimum level of energy that the battery can give to the microgrid
    def Bconstraint4_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_discharge[l, t] >= battery.soc_min * model.bd[l, t]
    model.Bconstraint4 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint4_rule)

    # Maixmum level of energy that can enter to the battery
    def Bconstraint5_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_charge[l, t] <= battery.soc_max * model.bc[l, t] 
    model.Bconstraint5 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint5_rule)

    # Maximum level of energy that the battery can give to the microgrid
    def Bconstraint6_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_discharge[l, t] <= battery.soc_max * model.bd[l, t] 
    model.Bconstraint6 = pyo.Constraint(model.BATTERIES, model.HTIME, rule=Bconstraint6_rule)
    
    # Charge and discharge control 
    def bcbd_rule(model,t):
      return  model.bc[t] + model.bd[t] <= 1
    model.bcbd_rule = pyo.Constraint(model.HTIME, rule=bcbd_rule)
   

    # Defines LPSP constraint
    def lpspcons_rule(model, t):
      if t >= (model.tlpsp - 1):
        rev = sum(model.d[t] for t in range((t-model.tlpsp+1), t+1)) 
        if rev > 0:
          return sum(model.s_minus[t] for t in range((t-model.tlpsp+1), t+1)) / sum(model.d[t] for t in range((t-model.tlpsp+1), t+1))  <= model.nse 
        else:
          return pyo.Constraint.Skip
      else:
        return pyo.Constraint.Skip
    model.lpspcons = pyo.Constraint(model.HTIME, rule=lpspcons_rule)


    
    
    #Objective funtion       

    # DefineTNPC operational constraint
    # Define TNPC operative constraint
    def tnpcop_rule(model):
        #TODO check cost unsupplied
        expr2 = 10*sum(model.s_minus[t] for t in model.HTIME)
        expr2 += sum(sum(generators_dict[k].va_op * model.p[k,t] for t in model.HTIME) for k in model.GENERATORS)
        return model.TNPC_OP == expr2
    model.tnpcop = pyo.Constraint(rule=tnpcop_rule)


    # Defines Objective function
    def obj2_rule(model):
      return ((model.TNPC + model.TNPC_OP) * model.CRF) / sum( model.d[t] for t in model.HTIME)
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)

    
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
    #s_status = results.solver.status
    term = {}
    # TODO: Check which other termination conditions may be interesting for us 
    # http://www.pyomo.org/blog/2015/1/8/accessing-solver
    #status {aborted, unknown}, Termination Condition {maxTimeLimit (put), locallyOptimal, maxEvaluations, licesingProblem}
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
        
        # Hourly data frame
        demand = pd.DataFrame(model.d.values(), columns=['demand'])
        
        generation = {k : [0]*len(model.HTIME) for k in model.GENERATORS}
        for (k,t), f in model.p.items():
          generation [k][t] = value(f)
        generation = pd.DataFrame(generation, columns=[*generation.keys()])
        
        # batery charge and discharge
        b_discharge_data = {l+'_b-' : [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.b_discharge.items():
          b_discharge_data [l+'_b-'][t] = value(f)
        b_discharge_df = pd.DataFrame(b_discharge_data, columns=[*b_discharge_data.keys()])

        b_charge_data = {l+'_b+': [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.b_charge.items():
          b_charge_data [l+'_b+'][t] = value(f)
        b_charge_df = pd.DataFrame(b_charge_data, columns=[*b_charge_data.keys()])
        
        soc_data = {l+'_soc' : [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.soc.items():
          soc_data [l+'_soc'][t] = value(f)
        soc_df = pd.DataFrame(soc_data, columns=[*soc_data.keys()])  
        
        # No supplied demand
        sminus_data = [0]*len(model.HTIME)
        lpsp_data = [0]*len(model.HTIME)
        for t in model.HTIME:
            sminus_data[t] = value(model.s_minus[t])
            if model.d[t] != 0:
              lpsp_data [t] = value(model.s_minus[t]) / value(model.d[t])
        
        sminus_df = pd.DataFrame(list(zip(sminus_data, lpsp_data)), columns = ['S-', 'LPSP'])
        

        self.df_results = pd.concat([demand, generation, b_discharge_df, b_charge_df, soc_df, sminus_df ], axis=1) 
        
        # general descriptives of the solution
        self.descriptive = {}
        
        # generators 
        generators = {}
        try:
            for k in model.GENERATORS:
               generators[k] = value(model.w[k])
            self.descriptive['generators'] = generators
        except:
            for k in model.GENERATORS:
                if generation[k].sum() > 0:
                    generators[k] = 1
            self.descriptive['generators'] = generators
        # technologies
        tecno_data = {}
        try:
            for i in model.TECHNOLOGIES:
               tecno_data[i] = value(model.y[i])
            self.descriptive['technologies'] = tecno_data
        except: #TODO
              a=1 
              
        bat_data = {}
        try: 
            for l in model.BATTERIES:
               bat_data[l] = value(model.q[l])
            self.descriptive['batteries'] = bat_data
        except:
            for l in model.BATTERIES:
                if b_discharge_df[l+'_b-'].sum() + b_charge_df[l+'_b+'].sum() > 0:
                    bat_data[l] = 1
            self.descriptive['batteries'] = bat_data 
                  
        
        brand_data = {}
        try:
            for (i, j) in model.TEC_BRAND:
                brand_data[i, j] = value(model.x[i,j])  
            self.descriptive['Brand'] = brand_data
        except:#TODO
            a=1 
        
        area = 0
        try:
            for k in model.GENERATORS:
                area += value(model.w[k]) * model.gen_area[k]          
            for l in model.BATTERIES:
              area += value(model.q[l]) * model.bat_area[l]
            self.descriptive['area'] = area
        except:
            for k in generators.keys():
                area += generators[k] * model.gen_area[k]          
            for l in bat_data.keys():
              area += bat_data[l] * model.bat_area[l]
            self.descriptive['area'] = area
            
            
        # objective function
        self.descriptive['LCOE'] = model.LCOE_value.expr()
        
        
        
        
    def generation_graph(self):
        bars = []
        for key, value in self.descriptive['generators'].items():
            if value==1:
                bars.append(go.Bar(name=key, x=self.df_results.index, y=self.df_results[key]))
        for key, value in self.descriptive['batteries'].items():
            if value==1:
                column_name = key+'_b-'
                bars.append(go.Bar(name=key, x=self.df_results.index, y=self.df_results[column_name]))
        
        #TODO poner gráfico de s menos
                bars.append(go.Bar(name='Unsupplied Demand',x=self.df_results.index, y=self.df_results['S-']))
                
        plot = go.Figure(data=bars)
        
        
        plot.add_trace(go.Scatter(x=self.df_results.index, y=self.df_results['demand'],
                    mode='lines',
                    name='Demand',
                    line=dict(color='grey', dash='dot')))
        
        self.df_results['b+'] = 0
        for key, value in self.descriptive['batteries'].items():
            if value==1:
                column_name = key+'_b+'
                self.df_results['b+'] += self.df_results[column_name]
        #self.df_results['Battery1_b+']+self.df_results['Battery2_b+']
        plot.add_trace(go.Bar(x=self.df_results.index, y=self.df_results['b+'],
                              base=-1*self.df_results['b+'],
                              marker_color='grey',
                              name='Charge'
                              ))
        
        # Set values y axis
        #plot.update_yaxes(range=[-self.df_results['b+'].max()-50, self.df_results['demand'].max()+200])
        #plot.update_yaxes(range=[-10, 30])
        # Change the bar mode
        plot.update_layout(barmode='stack')
        
        
        return plot
    
    

        
