# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:12 2022
@author: scastellanos
"""

import pyomo.environ as pyo
from pyomo.core import value
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
import pandas as pd 
import time
import copy


def make_model(generators_dict=None, 
               batteries_dict=None,  
               demand_df=None, 
               technologies_dict = None,  
               renewables_dict = None,
               amax = 0, 
               fuel_cost = 0,
               ir = 1, 
               nse = 0, 
               maxtec = 1, 
               mintec = 1,
               maxbr = 0,
               years = 1,
               w_cost = 0,
               tlpsp = 1,
               delta = 1):
    #generators_dict = dictionary of generators
    #batteries_dict = dictionary of batteries
    #demand_df = demand dataframe
    #technologies_dict = dictionary of technologies
    #renewables_dict = dictionary of renewable technologies
    #amax = maximum area
    #ir = interest rate
    #nse = maximum allowed not supplied energy
    #maxtec = maximum number of technologies allowed
    #mintec = minimum number of technologies allowed
    #maxbr= maximum brands allowed by each technology
    #w_cost = Penalized wasted energy cost
    #tlpsp = Number of lpsp periods for moving average
    #Delta = tax incentive

    
    # Sets
    model = pyo.ConcreteModel(name="Sizing microgrids")
    model.GENERATORS = pyo.Set(initialize=[k for k in generators_dict.keys()])
    model.BATTERIES = pyo.Set(initialize=[l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize=[i for i in technologies_dict.keys()])
    model.RENEWABLES = pyo.Set(initialize=[r for r in renewables_dict.keys()])
    model.TEC_BRAND = pyo.Set( initialize = [(i,j) for i in technologies_dict.keys() for j in technologies_dict[i]], ordered = False) #Indexed set - technologies / brands
    model.HTIME = pyo.Set(initialize=[t for t in range(len(demand_df))])

    # Parameters
    model.amax = pyo.Param(initialize=amax) #Maximum area
    model.fuel_cost = pyo.Param(initialize=fuel_cost) #Fuel Cost
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area for k in generators_dict.keys()})# Generator area
    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area for k in batteries_dict.keys()})# Battery area
    model.d = pyo.Param(model.HTIME, initialize = demand_df) #demand    
    model.ir = pyo.Param(initialize=ir) #Interest rate
    model.nse = pyo.Param(initialize=nse) #Available not supplied demand 
    model.maxtec = pyo.Param(initialize = maxtec) #Maximum technologies 
    model.mintec = pyo.Param(initialize = mintec) #Minimum technologies 
    model.maxbr = pyo.Param(model.TECHNOLOGIES, initialize = maxbr) #Maximum brand by each technology  
    model.t_years = pyo.Param(initialize = years) # Number of years evaluation of the project, for CRF
    CRF_calc = (model.ir * (1 + model.ir)**(model.t_years))/((1 + model.ir)**(model.t_years)-1) #CRF to LCOE
    model.CRF = pyo.Param(initialize = CRF_calc) #capital recovery factor 
    model.tlpsp = pyo.Param (initialize = tlpsp) #LPSP Time for moving average
    model.w_cost = pyo.Param (initialize = w_cost)
    model.delta = pyo.Param (initialize = delta)

    # Variables
    model.y = pyo.Var(model.TECHNOLOGIES, within=pyo.Binary) #select or not the technology
    model.x = pyo.Var(model.TEC_BRAND, within=pyo.Binary) #select or not the brand
    model.w = pyo.Var(model.GENERATORS, within=pyo.Binary) #select or not the generator
    model.q = pyo.Var(model.BATTERIES, within=pyo.Binary) #select or not the battery
    model.v = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.Binary) #select or not the generator in each period
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals) #Power generated
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals) #State of charge of the battery
    model.b_charge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals) #Network power to charge the battery
    model.b_discharge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals) #energy leaving from the battery and entering to the network
    model.s_minus = pyo.Var(model.HTIME, within=pyo.NonNegativeReals) #Not supplied energy
    model.s_plus = pyo.Var(model.HTIME, within=pyo.NonNegativeReals) #Wasted energy
    model.bd = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary) #Charge or not the battery in each period
    model.bc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary) #Disharge or not the battery in each period
    model.TNPC = pyo.Var(within=pyo.NonNegativeReals) #Total net present cost
    model.TNPC_OP = pyo.Var(within=pyo.NonNegativeReals) #Operative cost
    model.operative_cost = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals)
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
    def G_rule (model, k, t):
      gen = generators_dict[k]
      if gen.tec == 'D':
          return model.p[k,t] <= gen.DG_max * model.v[k,t]
      else:
          return model.p[k,t] == gen.gen_rule[t] * model.v[k,t]
    model.G_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_rule)
    
    # Minimum Diesel  
    def G_mindiesel (model, k, t):
      gen = generators_dict[k]
      if gen.tec == 'D':
          return model.p[k,t] >= gen.DG_min * model.v[k,t]
      else:
          return pyo.Constraint.Skip
    model.G_mindiesel = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_mindiesel)
    
    # Defines balance rule
    def balance_rule(model, t):
      return  sum(model.p[(k,t)] for k in model.GENERATORS) +  sum(model.b_discharge[(l,t)] for l in model.BATTERIES) + model.s_minus[t] == model.s_plus[t] + model.d[t]  + sum(model.b_charge[(l,t)] for l in model.BATTERIES)
    model.balance_rule = pyo.Constraint(model.HTIME, rule=balance_rule)

    # Defines operative cost
    def cop_rule(model,k, t):
      gen = generators_dict[k]
      if gen.tec == 'D': 
          return model.operative_cost[k,t] == (gen.f0 * gen.DG_max  + gen.f1 * model.p[k,t])*model.fuel_cost* model.w[k]
      else:
          return model.operative_cost[k,t] == gen.cost_vopm * model.p[k,t] 
    model.cop_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cop_rule)
        
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
    
    # Defines constraint of minimum number of technologies
    def mintec_rule(model):
      return sum(model.y[i] for i in model.TECHNOLOGIES) >= model.mintec
    model.mintec_rule = pyo.Constraint(rule=mintec_rule)

    # Defines constraint of maximum number of brands
    def maxbr_rule (model, i):
        return sum (model.x[i,j] for  j in technologies_dict[i]) <= model.maxbr[i]
    model.maxbr_rule = pyo.Constraint(model.TECHNOLOGIES, rule = maxbr_rule)
    
    
     # Defines rule relation Solar - Diesel
    def dieselsolar_rule(model,k,t):
        gen = generators_dict[k]
        if gen.tec == 'S':
            return sum( model.v[k1,t] for k1 in model.GENERATORS if generators_dict[k1].tec == 'D') >= model.v[k,t]
        else:
            return pyo.Constraint.Skip
    #model.dieselsolar_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=dieselsolar_rule)

    #If select a technology must to install at least one generator or battery
    def wqy_rule(model, i):
        if i != 'B':
            return sum(model.w[k] for k in model.GENERATORS if generators_dict[k].tec == i) >= model.y[i] 
        else:
            return   sum(model.q[l] for l in model.BATTERIES) >= model.y[i]
    model.wqy_rule = pyo.Constraint(model.TECHNOLOGIES, rule=wqy_rule)



    #Objective function
        
    # Defines TNPC constraint
    def tnpcc_rule(model): 
            expr =  sum(batteries_dict[l].cost_up * model.delta * model.q[l] for l in model.BATTERIES) 
            expr += sum(generators_dict[k].cost_up*model.w[k] for k in model.GENERATORS if generators_dict[k].tec == 'D')
            expr += sum(generators_dict[k].cost_up* model.delta *model.w[k] for k in model.GENERATORS if generators_dict[k].tec != 'D')
            expr += sum(generators_dict[k].cost_r*model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_r * model.q[l]  for l in model.BATTERIES) 
            expr -= (sum(generators_dict[k].cost_s*model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_s * model.q[l]  for l in model.BATTERIES))
            return model.TNPC == expr
    model.tnpcc = pyo.Constraint(rule=tnpcc_rule)

    # Define TNPC operative constraint
    def tnpcop_rule(model):
        expr2 = sum(sum(model.operative_cost[k,t] for t in model.HTIME) for k in model.GENERATORS)
        expr2 += sum(generators_dict[k].cost_fopm*model.w[k] for k in model.GENERATORS) + sum(batteries_dict[l].cost_fopm * model.q[l]  for l in model.BATTERIES)
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

    #control that s- is lower than demand if tlpsp is two or more
    def other_rule(model, t):
        if (model.tlpsp > 1):
            return model.s_minus[t] <= model.d[t]
        else:
            return pyo.Constraint.Skip
    model.other_rule = pyo.Constraint(model.HTIME, rule=other_rule)
    
    # Defines objective function
    def obj2_rule(model):
      return ((model.TNPC * model.CRF + model.TNPC_OP) / sum( model.d[t] for t in model.HTIME)) +  model.w_cost * sum( model.s_plus[t] for t in model.HTIME) 
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)
    
    return model



def make_model_operational(generators_dict=None, 
               batteries_dict=None,  
               demand_df=None, 
               technologies_dict = None,  
               renewables_dict = None,     
               fuel_cost = 0,
               nse = 0, 
               TNPCCRF = 0,
               w_cost = 0,
               tlpsp = 1):
    #generators_dict = dictionary of generators - input (Sizing decision)
    #batteries_dict = dictionary of batteries - input (Sizing decision)
    #demand_df = demand dataframe
    #technologies_dict = dictionary of technologies - input (Sizing decision)
    #renewables_dict = dictionary of renewable technologies - input (Sizing decision)
    #nse = maximum allowed not supplied energy
    #TNPC = Total net present cost - input (Sizing decision)
    #CRF = Capital recovery factor - input (Sizing decision)
    #tlpsp = Number of lpsp periods for moving average
    #lpsp_cost = cost of unsupplied energy

        
    model = pyo.ConcreteModel(name="Sizing microgrids Operational")
    
    # Sets
    model.GENERATORS = pyo.Set(initialize=[k for k in generators_dict.keys()])
    model.BATTERIES = pyo.Set(initialize=[l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize=[i for i in technologies_dict.keys()])
    model.RENEWABLES = pyo.Set(initialize=[r for r in renewables_dict.keys()])
    model.TEC_BRAND = pyo.Set( initialize = [(i,j) for i in technologies_dict.keys() for j in technologies_dict[i]], ordered = False) #Index set - technologies / brand
    model.HTIME = pyo.Set(initialize=[t for t in range(len(demand_df))])

    # Parameters 
    model.d = pyo.Param(model.HTIME, initialize = demand_df) #demand     
    model.fuel_cost = pyo.Param(initialize=fuel_cost) #Fuel Cost
    model.nse = pyo.Param(initialize=nse) # Available unsupplied demand  
    model.TNPCCRF = pyo.Param(initialize = TNPCCRF) #Total net present cost (Sizing decision)
    model.tlpsp = pyo.Param (initialize = tlpsp) #LPSP for moving average
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area for k in generators_dict.keys()})# Generator area
    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area for k in batteries_dict.keys()})# Battery area
    model.w_cost = pyo.Param (initialize = w_cost)

    # Variables
    model.v = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.Binary) #select or not the generator in each period
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals) #Power generated
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals) #State of charge of the battery
    model.b_charge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals) #Network power to charge the battery
    model.b_discharge = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.NonNegativeReals) #energy leaving from the battery and entering to the network
    model.s_minus = pyo.Var(model.HTIME, within=pyo.NonNegativeReals) #Not supplied energy
    model.s_plus = pyo.Var(model.HTIME, within=pyo.NonNegativeReals) #Wasted energy
    model.bd = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary) #Charge or not the battery in each period
    model.bc = pyo.Var(model.BATTERIES, model.HTIME, within=pyo.Binary) #Disharge or not the battery in each period
    model.TNPC_OP = pyo.Var(within=pyo.NonNegativeReals) #Operative cost
    model.operative_cost = pyo.Var(model.GENERATORS, model.HTIME, within=pyo.NonNegativeReals)
    # Constraints

    # Generation rule    
    def G_rule (model, k, t):
      gen = generators_dict[k]
      if gen.tec == 'D':
          return model.p[k,t] <= gen.DG_max * model.v[k,t]
      else:
          return model.p[k,t] == gen.gen_rule[t] * model.v[k,t]
    model.G_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_rule)
    
    # Minimum Diesel  
    def G_mindiesel (model, k, t):
      gen = generators_dict[k]
      if gen.tec == 'D':
          return model.p[k,t] >= gen.DG_min * model.v[k,t]
      else:
          return pyo.Constraint.Skip
    model.G_mindiesel = pyo.Constraint(model.GENERATORS, model.HTIME, rule=G_mindiesel)
    
    # Defines energy balance
    def balance_rule(model, t):
      return  sum(model.p[(k,t)] for k in model.GENERATORS) +  sum(model.b_discharge[(l,t)] for l in model.BATTERIES) + model.s_minus[t] ==  model.s_plus[t] + model.d[t]  + sum(model.b_charge[(l,t)] for l in model.BATTERIES)
    model.balance_rule = pyo.Constraint(model.HTIME, rule=balance_rule)

    # Defines operative cost
    def cop_rule(model,k, t):
      gen = generators_dict[k]
      if gen.tec == 'D': 
          return model.operative_cost[k,t] == (gen.f0 * gen.DG_max  + gen.f1 *  model.p[k,t])*model.fuel_cost
      else:
          return model.operative_cost[k,t] == gen.cost_vopm * model.p[k,t] 
    model.cop_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=cop_rule)
    
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
    def bcbd_rule(model, l, t):
      return  model.bc[l, t] + model.bd[l, t] <= 1
    model.bcbd_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule=bcbd_rule)
   

    
    # Defines rule relation Solar - Diesel
    def dieselsolar_rule(model,k,t):
        gen = generators_dict[k]
        if gen.tec == 'S':
            return sum( model.v[k1,t] for k1 in model.GENERATORS if generators_dict[k1].tec == 'D') >= model.v[k,t]
        else:
            return pyo.Constraint.Skip
    #model.dieselsolar_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule=dieselsolar_rule)


    # Define TNPC operative constraint
    def tnpcop_rule(model):
        expr2 = sum(sum(model.operative_cost[k,t] for t in model.HTIME) for k in model.GENERATORS)
        return model.TNPC_OP == expr2
    model.tnpcop = pyo.Constraint(rule=tnpcop_rule)


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
    
    #control that s- is lower than demand if tlpsp is two or more
    def other_rule(model, t):
        if (model.tlpsp > 1):
            return model.s_minus[t] <= model.d[t]
        else:
            return pyo.Constraint.Skip
    model.other_rule = pyo.Constraint(model.HTIME, rule=other_rule)


    # Defines Objective function
    def obj2_rule(model):
      return ((model.TNPCCRF + model.TNPC_OP) / sum( model.d[t] for t in model.HTIME)) +  model.w_cost * sum( model.s_plus[t] for t in model.HTIME) 
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule=obj2_rule)
    

    
    return model



def solve_model(model,
                optimizer='gurobi',
                mipgap=0.02,
                tee=True):
    solver = pyo.SolverFactory(optimizer)
    solver.options['MIPGap'] = mipgap
    timea = time.time() #initial time
    results = solver.solve(model, tee = tee)
    term_cond = results.solver.termination_condition #termination condition
    #s_status = results.solver.status
    term = {}
    # http://www.pyomo.org/blog/2015/1/8/accessing-solver
    #status {aborted, unknown}, Termination Condition {maxTimeLimit (put), locallyOptimal, maxEvaluations, licesingProblem}
    if term_cond != pyo.TerminationCondition.optimal:
          term['Temination Condition'] = format(term_cond)
          term['LCOE'] = None 
          term['LPSP'] = value(model.nse)
          execution_time = time.time() - timea #final time
          term['Execution time'] = execution_time
          #raise RuntimeError("Optimization failed.")

    else: 
          term['Temination Condition'] = format(term_cond)
          term['LCOE'] = model.LCOE_value.expr()
          term['LPSP'] = value(model.nse)
          execution_time = time.time() - timea
          term['Execution time'] = execution_time    
    return results, term


class Results():
    def __init__(self, model):
        
        # Hourly data frame
        demand = pd.DataFrame(model.d.values(), columns=['demand'])
        
        #Generator data frame
        generation = {k : [0]*len(model.HTIME) for k in model.GENERATORS}
        for (k,t), f in model.p.items():
          generation [k][t] = value(f)
          
        generation = pd.DataFrame(generation, columns=[*generation.keys()])
               
        #Operative cost data frame
        generation_cost_data = {k+'_cost' : [0]*len(model.HTIME) for k in model.GENERATORS}
        for (k,t), f in model.operative_cost.items():
          generation_cost_data [k+'_cost'][t] = value(f)
        generation_cost = pd.DataFrame(generation_cost_data, columns=[*generation_cost_data.keys()])
        
        # batery charge and discharge
        b_discharge_data = {l+'_b-' : [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.b_discharge.items():
          b_discharge_data [l+'_b-'][t] = value(f)
        
        b_discharge_df = pd.DataFrame(b_discharge_data, columns=[*b_discharge_data.keys()])
        
        b_charge_data = {l+'_b+': [0]*len(model.HTIME) for l in model.BATTERIES}
        for (l,t), f in model.b_charge.items():
          b_charge_data [l+'_b+'][t] = value(f)
        b_charge_df = pd.DataFrame(b_charge_data, columns=[*b_charge_data.keys()])
        
        #SOC Battery dataframe
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


        # Wasted Energy
        splus_data = [0]*len(model.HTIME)
        for t in model.HTIME:
            splus_data[t] = value(model.s_plus[t])
            
        splus_df = pd.DataFrame(splus_data, columns = ['Wasted Energy'])
                

        self.df_results = pd.concat([demand, generation, b_discharge_df, b_charge_df, soc_df, sminus_df, splus_df, generation_cost], axis=1) 
        
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

