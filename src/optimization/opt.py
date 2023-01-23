# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:12 2022
@author: scastellanos
"""

import pyomo.environ as pyo
from pyomo.core import value
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import pandas as pd 
import time
import copy


def make_model(generators_dict = None, 
               batteries_dict = None,  
               demand_df = None, 
               technologies_dict = None,  
               renewables_dict = None,
               amax = 0, 
               fuel_cost = 0,
               ir = 1, 
               nse = 0, 
               years = 1,
               splus_cost = 0,
               tlpsp = 1,
               delta = 1,
               inverter = 0,
               nse_cost = {
                   "L1":[0.015,0],
                   "L2":[0.05,0],
                   "L3":[0.9, 0],
                   "L4":[1,0]
                   } 
               ):
    '''
    make pyomo model- one stage approach
    
    PARAMETERS
    ----------
    generators_dict = dictionary of generators
    batteries_dict = dictionary of batteries
    demand_df = demand dataframe
    technologies_dict = dictionary of technologies
    renewables_dict = dictionary of renewable technologies
    amax = maximum area
    ir = interest rate
    nse = maximum allowed not supplied energy
    maxtec = maximum number of technologies allowed
    mintec = minimum number of technologies allowed
    maxbr= maximum brands allowed by each technology
    splus_cost = Penalized surplus energy cost
    tlpsp = Number of lpsp periods for moving average
    delta = tax incentive
    inverter = inverter cost
    nse_cost = cost to calcalte not supplied load
    
    RETURN
    -------
    Model
    
    '''
    # Sets
    model = pyo.ConcreteModel(name = "Sizing microgrids - One Stage")
    model.GENERATORS = pyo.Set(initialize = [k for k in generators_dict.keys()])
    model.GENERATORS_DIESEL = pyo.Set(initialize = [k for k in model.GENERATORS 
                                                  if generators_dict[k].tec == 'D'])
    model.GENERATORS_REN = pyo.Set(initialize = [k for k in model.GENERATORS 
                                               if generators_dict[k].tec != 'D'])

    model.BATTERIES = pyo.Set(initialize = [l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize = [i for i in technologies_dict.keys()])
    model.RENEWABLES = pyo.Set(initialize = [r for r in renewables_dict.keys()])
    model.HTIME = pyo.Set(initialize = [t for t in range(len(demand_df))])

    # Parameters
    model.amax = pyo.Param(initialize =amax) #Maximum area
    model.fuel_cost = pyo.Param(initialize =fuel_cost) #Fuel Cost
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area
                                                               for k in generators_dict.keys()})# Generator area

    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area 
                                                              for k in batteries_dict.keys()})# Battery area
    
    model.demand = pyo.Param(model.HTIME, initialize = demand_df) #demand    
    model.ir = pyo.Param(initialize = ir) #Interest rate
    model.nse = pyo.Param(initialize = nse) #Available not supplied demand  
    # Number of years evaluation of the project, for CRF
    model.t_years = pyo.Param(initialize = years) 
    crf_calc = ((model.ir * (1 + model.ir) ** (model.t_years))
                /((1 + model.ir) ** (model.t_years) - 1)) #CRF to LCOE
    model.CRF = pyo.Param(initialize = crf_calc) #capital recovery factor 
    model.tlpsp = pyo.Param (initialize = tlpsp) #LPSP Time for moving average
    model.splus_cost = pyo.Param (initialize = splus_cost)
    model.delta = pyo.Param (initialize = delta)
    model.inverter = pyo.Param (initialize = inverter)
    #parameters to piecewise function
    model.x1_cost = pyo.Param (initialize = nse_cost["L1"][1])
    model.x2_cost = pyo.Param (initialize = nse_cost["L2"][1])
    model.x3_cost = pyo.Param (initialize = nse_cost["L3"][1])
    model.x4_cost = pyo.Param (initialize = nse_cost["L4"][1])
    model.x1_limit = pyo.Param (initialize = nse_cost["L1"][0])
    model.x2_limit = pyo.Param (initialize = nse_cost["L2"][0])
    model.x3_limit = pyo.Param (initialize = nse_cost["L3"][0])
    model.x4_limit = pyo.Param (initialize = nse_cost["L4"][0])
    # Variables
    model.w = pyo.Var(model.GENERATORS, within = pyo.Binary) #select or not the generator
    model.q = pyo.Var(model.BATTERIES, within = pyo.Binary) #select or not the battery
    #select or not the generator in each period
    model.v = pyo.Var(model.GENERATORS_DIESEL, model.HTIME, within = pyo.Binary) 
    #Power generated
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within = pyo.NonNegativeReals)
    #State of charge of the battery
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.NonNegativeReals) 
    #Network power to charge the battery
    model.b_charge = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.NonNegativeReals) 
    #energy leaving from the battery and entering to the network
    model.b_discharge = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.NonNegativeReals) 
    #Not supplied energy
    model.s_minus = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    #Wasted energy
    model.s_plus = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    #Charge or not the battery in each period
    model.bd = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.Binary) 
    #Disharge or not the battery in each period
    model.bc = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.Binary) 
    model.operative_cost = pyo.Var(model.GENERATORS_DIESEL, model.HTIME, within = pyo.NonNegativeReals)
    
    #calculate not supplied cost piecewise
    model.x1 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    model.x2 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    model.x3 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    model.x4 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    #calculate not supplied cost piecewise
    model.z1 = pyo.Var(model.HTIME, within = pyo.Binary) 
    model.z2 = pyo.Var(model.HTIME, within = pyo.Binary) 
    model.z3 = pyo.Var(model.HTIME, within = pyo.Binary) 
    # Constraints

    # Defines area rule
    def area_rule(model):
      return  (sum(model.gen_area[k] * model.w[k] for k in model.GENERATORS) 
               + sum(model.bat_area[l]* model.q[l] for l in model.BATTERIES)) <= model.amax
    model.area_rule = pyo.Constraint(rule = area_rule)

    # Defines rule to activate or deactivate generators for each period of time
    def vkt_rule(model,k, t):
        return model.v[k, t] <= model.w[k]
    model.vkt_rule = pyo.Constraint(model.GENERATORS_DIESEL, 
                                    model.HTIME, rule = vkt_rule)

    # Generation rule    
    def G_rule (model, k, t):
      gen = generators_dict[k]
      if gen.tec == 'D':
          return model.p[k, t] <= gen.DG_max * model.v[k, t]
      else:
          return model.p[k, t] == gen.gen_rule[t]  * model.w[k]
    model.G_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule = G_rule)
    
    # Minimum Diesel  
    def G_min_diesel (model, k, t):
      gen = generators_dict[k]
      return model.p[k, t] >= gen.DG_min * model.v[k, t]
    model.G_min_diesel = pyo.Constraint(model.GENERATORS_DIESEL, 
                                       model.HTIME, rule = G_min_diesel)
    
    # Defines balance rule
    def balance_rule(model, t):
      return  (sum(model.p[(k, t)] for k in model.GENERATORS) 
               +  sum(model.b_discharge[(l, t)] for l in model.BATTERIES) 
               + model.s_minus[t] == model.s_plus[t] + model.demand[t]  
               + sum(model.b_charge[(l, t)] for l in model.BATTERIES))
    model.balance_rule = pyo.Constraint(model.HTIME, rule = balance_rule)

    # Defines operative cost
    def cost_op_rule(model,k, t):
      gen = generators_dict[k]
      return model.operative_cost[k, t] == (gen.f0 
                                           * gen.DG_max * model.v[k, t]  + gen.f1 * model.p[k, t]) * model.fuel_cost
    model.cop_rule = pyo.Constraint(model.GENERATORS_DIESEL, model.HTIME, rule = cost_op_rule)
        
    #Batteries management
    def soc_rule(model, l, t):
      battery = batteries_dict[l]
      if t == 0:
          expr =  model.q[l] * battery.eb_zero * (1 - battery.alpha)
          expr += model.b_charge[l, t] * battery.efc
          expr -= (model.b_discharge[l, t] / battery.efd)
          return model.soc[l, t] == expr 
      else:
          expr = model.soc[l, t-1] * (1 - battery.alpha)
          expr += model.b_charge[l, t] * battery.efc
          expr -= (model.b_discharge[l, t] / battery.efd)
          return model.soc[l, t] == expr
    model.soc_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule = soc_rule)

    #Highest level of SOC
    def Bconstraint_rule(model, l, t):
        battery = batteries_dict[l]
        return model.soc[l, t] <= battery.soc_max * model.q[l]
    model.Bconstraint = pyo.Constraint(model.BATTERIES, 
                                       model.HTIME, rule = Bconstraint_rule)

    #Minimun SOC level
    def Bconstraint2_rule(model, l, t):
      battery = batteries_dict[l]
      return model.soc[l, t] >= battery.soc_min * model.q[l]
    model.Bconstraint2 = pyo.Constraint(model.BATTERIES, 
                                        model.HTIME, rule = Bconstraint2_rule)

    #Maximum level of energy that can enter to the battery
    def Bconstraint3_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_charge[l, t] <= battery.soc_max * model.bc[l, t] 
    model.Bconstraint3 = pyo.Constraint(model.BATTERIES, 
                                        model.HTIME, rule = Bconstraint3_rule)

    #Maximum level of energy that the battery can give to the microgrid
    def Bconstraint4_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_discharge[l, t] <= battery.soc_max * model.bd[l, t] 
    model.Bconstraint4 = pyo.Constraint(model.BATTERIES, 
                                        model.HTIME, rule = Bconstraint4_rule)
    
    # Charge and discharge control 
    def bcbd_rule(model, l, t):
      return  model.bc[l, t] + model.bd[l, t] <= 1
    model.bcbd_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule = bcbd_rule)
    

    # Defines LPSP constraint
    def lpsp_cons_rule(model, t):
      if t >= (model.tlpsp - 1):
        rev_mov_average = sum(model.demand[t] for t in range((t - model.tlpsp + 1), t + 1)) 
        if rev_mov_average > 0:
          return sum(model.s_minus[t] for t in range((t - model.tlpsp + 1)
                                                     , t + 1)) / rev_mov_average  <= model.nse 
        else:
          return pyo.Constraint.Skip
      else:
        return pyo.Constraint.Skip
    model.lpsp_cons = pyo.Constraint(model.HTIME, rule = lpsp_cons_rule)

    #control that s- is lower than demand if tlpsp is two or more
    def additional_rule(model, t):
        if (model.tlpsp > 1):
            return model.s_minus[t] <= model.demand[t]
        else:
            return pyo.Constraint.Skip
    model.additional_rule = pyo.Constraint(model.HTIME, rule = additional_rule)
    
    # Defines strategic rule relation Renewable - Diesel
    def relation_rule(model):
            expr = sum( model.w[k] for k in model.GENERATORS_REN) 
            expr2 = sum( model.w[k] for k in model.GENERATORS_DIESEL)
            expr3 = sum(model.q[l] for l in model.BATTERIES)
            expr4 = len(model.GENERATORS)
            return  expr <= expr4 * (expr2 + expr3)
    model.relation_rule = pyo.Constraint(rule = relation_rule)

    
    # Defines operational rule relation Renewable - Diesel
    def op_relation_rule(model, t):
        check_tec = 0
        for i in generators_dict.values():
            if (i.tec == 'S' or i.tec == 'W'):
                check_tec = 1
                break
        if (check_tec == 1):
            expr4 = sum(model.demand[t1] for t1 in model.HTIME)                 
            return sum( model.p[k, t] for k in model.GENERATORS_REN) <= expr4 * (sum( model.v[k2,t]
                                                                                     for k2 in model.GENERATORS_DIESEL)  + sum(model.soc[l, t] for l in model.BATTERIES)) 
        else:
            return pyo.Constraint.Skip
    model.op_relation_rule = pyo.Constraint(model.HTIME, rule = op_relation_rule)

    
    #Constraints s- cost piecewise
    def sminusx_rule(model, t):
        return model.s_minus[t] == model.x1[t] + model.x2[t] + model.x3[t] + model.x4[t]
    model.sminusx_rule = pyo.Constraint(model.HTIME, rule = sminusx_rule)
    
    def sminusx1_rule(model, t):
        return model.x1[t] <= model.x1_limit * model.demand[t]
    model.sminusx1_rule = pyo.Constraint(model.HTIME, rule = sminusx1_rule)
    
    def sminusx2_rule(model, t):
        return model.x1[t] >= model.x1_limit * model.demand[t] * model.z1[t]
    model.sminusx2_rule = pyo.Constraint(model.HTIME, rule = sminusx2_rule)    

    def sminusx3_rule(model, t):
        return model.x2[t] <= (model.x2_limit - model.x1_limit) * model.demand[t] * model.z1[t]
    model.sminusx3_rule = pyo.Constraint(model.HTIME, rule = sminusx3_rule)
    
    def sminusx4_rule(model, t):
        return model.x2[t] >= (model.x2_limit - model.x1_limit) * model.demand[t] * model.z2[t]
    model.sminusx4_rule = pyo.Constraint(model.HTIME, rule = sminusx4_rule)  
    
    def sminusx5_rule(model, t):
        return model.x3[t] <= (model.x3_limit - model.x2_limit) * model.demand[t] * model.z2[t]
    model.sminusx5_rule = pyo.Constraint(model.HTIME, rule = sminusx5_rule)
    
    def sminusx6_rule(model, t):
        return model.x3[t] >= (model.x3_limit - model.x2_limit) * model.demand[t] * model.z3[t]
    model.sminusx6_rule = pyo.Constraint(model.HTIME, rule = sminusx6_rule)  

    def sminusx7_rule(model, t):
        return model.x4[t] <= (model.x4_limit - model.x3_limit) * model.demand[t] * model.z3[t]
    model.sminusx7_rule = pyo.Constraint(model.HTIME, rule = sminusx7_rule)
     
    #Objective function
    def obj_rule(model):
        tnpc =  sum(batteries_dict[l].cost_up * model.delta
                    * model.q[l] for l in model.BATTERIES)
        
        tnpc += sum(generators_dict[k].cost_up * model.w[k]
                    for k in model.GENERATORS_DIESEL)
        
        tnpc += sum(generators_dict[k].cost_up * model.delta * model.w[k]
                    for k in model.GENERATORS_REN)
        
        tnpc +=  sum(batteries_dict[l].cost_r * model.delta
                     * model.q[l] for l in model.BATTERIES) 
        
        tnpc += sum(generators_dict[k].cost_r * model.w[k] 
                    for k in model.GENERATORS_DIESEL)
        
        tnpc += sum(generators_dict[k].cost_r * model.delta * model.w[k] 
                    for k in model.GENERATORS_REN)
        
        tnpc -= (sum(generators_dict[k].cost_s * model.w[k] for k in model.GENERATORS) 
                 + sum(batteries_dict[l].cost_s * model.q[l]  for l in model.BATTERIES))
        tnpc += sum(generators_dict[k].cost_fopm * model.w[k] 
                    for k in model.GENERATORS) + sum(batteries_dict[l].cost_fopm * model.q[l]  
                                                     for l in model.BATTERIES)
        
        #sum inverter cost
        tnpc += model.inverter
        #tnpc += sum(generators_dict[k].DG_max* model.inverter * model.w[k] for k in model.GENERATORS if generators_dict[k].tec == 'D')
        #tnpc += sum(generators_dict[k].Ppv_stc* model.inverter * model.w[k] for k in model.GENERATORS if generators_dict[k].tec == 'S')
        #tnpc += sum(generators_dict[k].P_y* model.inverter * model.w[k] for k in model.GENERATORS if generators_dict[k].tec == 'W')
        #tnpc += sum(batteries_dict[k].soc_max* model.inverter * model.q[l] for l in model.BATTERIES)
        
        tnpc_op = sum(sum(model.operative_cost[k, t] for t in model.HTIME) 
                      for k in model.GENERATORS_DIESEL)
        
        tnpc_op += sum(generators_dict[k].cost_rule * model.w[k] for k in model.GENERATORS_REN)
        
        tnpc_op += sum(sum(model.b_discharge[l, t] * batteries_dict[l].cost_vopm
                           for l in model.BATTERIES) for t in model.HTIME)
        lcoe_cost = tnpc * model.CRF + tnpc_op 
        lcoe_cost += model.splus_cost * sum( model.s_plus[t] for t in model.HTIME)
        #s- cost piecewise
        lcoe_cost += sum(model.x1_cost * model.x1[t] for t in model.HTIME)
        lcoe_cost += sum(model.x2_cost * model.x2[t] for t in model.HTIME)
        lcoe_cost += sum(model.x3_cost * model.x3[t] for t in model.HTIME)
        lcoe_cost += sum(model.x4_cost * model.x4[t] for t in model.HTIME)
        lcoe = lcoe_cost/ (sum( model.demand[t] for t in model.HTIME))
        return lcoe
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule = obj_rule)
    '''
    #ambiental cost
    model.ambiental = pyo.Param (initialize = ambiental)
    model.ambiental_turn_on = pyo.Param (initialize = ambiental_turn_on)
    # Defines operative cost
    def cop_rule(model,k, t):
      gen = generators_dict[k]
      aux = (gen.f0 * gen.DG_max * model.v[k, t] + gen.f1 * model.p[k, t] ) * model.fuel_cost
      aux2 = model.v[k, t] * model.ambiental_turn_on + model.p[k, t] * model.ambiental
      return model.operative_cost[k, t] == aux + aux2
    model.cop_rule = pyo.Constraint(model.GENERATORS_DIESEL, model.HTIME, rule = cop_rule)
    '''
    
    return model


def make_model_operational(generators_dict = None, 
               batteries_dict = None,  
               demand_df = None, 
               technologies_dict = None,  
               renewables_dict = None,     
               fuel_cost = 0,
               nse = 0, 
               TNPCCRF = 0,
               splus_cost = 0,
               tlpsp = 1,
               nse_cost = {"L1":[0.015,0],
                              "L2":[0.05,0],
                              "L3":[0.9, 0],
                              "L4":[1,0]}):
    '''
    Generates the model (operational phase) - two stage approach 
    
    PARAMETERS
    ----------
    generators_dict = dictionary of generators - input (Sizing decision)
    batteries_dict = dictionary of batteries - input (Sizing decision)
    demand_df = demand dataframe
    technologies_dict = dictionary of technologies - input (Sizing decision)
    renewables_dict = dictionary of renewable technologies - input (Sizing decision)
    nse = maximum allowed not supplied energy
    TNPC = Total net present cost - input (Sizing decision)
    CRF = Capital recovery factor - input (Sizing decision)
    tlpsp = Number of lpsp periods for moving average
    lpsp_cost = cost of unsupplied energy
    splus_cost = Penalized surplus energy cost
    nse_cost = cost to calcalte not supplied load
    
    RETURNS
    -------
    Model
    '''

        
    model = pyo.ConcreteModel(name = "Sizing microgrids Operational")
    
    # Sets
    model.GENERATORS = pyo.Set(initialize = [k for k in generators_dict.keys()])
    model.GENERATORS_DIESEL = pyo.Set(initialize = [k for k in model.GENERATORS 
                                                  if generators_dict[k].tec == 'D'])
    model.GENERATORS_REN = pyo.Set(initialize = [k for k in model.GENERATORS 
                                               if generators_dict[k].tec != 'D'])
    model.BATTERIES = pyo.Set(initialize = [l for l in batteries_dict.keys()])
    model.TECHNOLOGIES = pyo.Set(initialize = [i for i in technologies_dict.keys()])
    model.RENEWABLES = pyo.Set(initialize = [r for r in renewables_dict.keys()])
    model.HTIME = pyo.Set(initialize = [t for t in range(len(demand_df))])

    # Parameters 
    model.demand = pyo.Param(model.HTIME, initialize = demand_df) #demand     
    model.fuel_cost = pyo.Param(initialize = fuel_cost) #Fuel Cost
    model.nse = pyo.Param(initialize = nse) # Available unsupplied demand  
    #Total net present cost (Sizing decision)
    model.tnpccrf = pyo.Param(initialize = TNPCCRF) 
    model.tlpsp = pyo.Param (initialize = tlpsp) #LPSP for moving average
    model.gen_area = pyo.Param(model.GENERATORS, initialize = {k:generators_dict[k].area
                                                               for k in generators_dict.keys()})# Generator area
    model.bat_area = pyo.Param(model.BATTERIES, initialize = {k:batteries_dict[k].area 
                                                              for k in batteries_dict.keys()})# Battery area
    model.splus_cost = pyo.Param (initialize = splus_cost)
    #Data from piecewise function
    model.x1_cost = pyo.Param (initialize = nse_cost["L1"][1])
    model.x2_cost = pyo.Param (initialize = nse_cost["L2"][1])
    model.x3_cost = pyo.Param (initialize = nse_cost["L3"][1])
    model.x4_cost = pyo.Param (initialize = nse_cost["L4"][1])
    model.x1_limit = pyo.Param (initialize = nse_cost["L1"][0])
    model.x2_limit = pyo.Param (initialize = nse_cost["L2"][0])
    model.x3_limit = pyo.Param (initialize = nse_cost["L3"][0])
    model.x4_limit = pyo.Param (initialize = nse_cost["L4"][0])

    # Variables
    #select or not the generator in each period
    model.v = pyo.Var(model.GENERATORS_DIESEL, model.HTIME, within = pyo.Binary) 
    #Power generated
    model.p = pyo.Var(model.GENERATORS, model.HTIME, within = pyo.NonNegativeReals)
    #State of charge of the battery
    model.soc = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.NonNegativeReals) 
    #Network power to charge the battery
    model.b_charge = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.NonNegativeReals) 
    #energy leaving from the battery and entering to the network
    model.b_discharge = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.NonNegativeReals) 
    #Not supplied energy
    model.s_minus = pyo.Var(model.HTIME, within = pyo.NonNegativeReals)
    #Wasted energy
    model.s_plus = pyo.Var(model.HTIME, within = pyo.NonNegativeReals)
    #Charge or not the battery in each period
    model.bd = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.Binary)
    #Disharge or not the battery in each period
    model.bc = pyo.Var(model.BATTERIES, model.HTIME, within = pyo.Binary) 
    model.operative_cost = pyo.Var(model.GENERATORS_DIESEL, model.HTIME, within = pyo.NonNegativeReals)
    #calculate not supplied cost piecewise
    model.x1 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals) 
    model.x2 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals)
    model.x3 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals)
    model.x4 = pyo.Var(model.HTIME, within = pyo.NonNegativeReals)
    #calculate not supplied cost piecewise
    model.z1 = pyo.Var(model.HTIME, within = pyo.Binary) 
    model.z2 = pyo.Var(model.HTIME, within = pyo.Binary) 
    model.z3 = pyo.Var(model.HTIME, within = pyo.Binary) 
    
    # Constraints

    # Generation rule    
    def G_rule (model, k, t):
      gen = generators_dict[k]
      if gen.tec == 'D':
          return model.p[k, t] <= gen.DG_max * model.v[k, t]
      else:
          return model.p[k, t] == gen.gen_rule[t] 
    model.G_rule = pyo.Constraint(model.GENERATORS, model.HTIME, rule = G_rule)
    
    # Minimum Diesel  
    def G_min_diesel (model, k, t):
      gen = generators_dict[k]
      return model.p[k, t] >= gen.DG_min * model.v[k, t]
      
    model.G_min_diesel = pyo.Constraint(model.GENERATORS_DIESEL,
                                       model.HTIME, rule = G_min_diesel)
    
    # Defines energy balance
    def balance_rule(model, t):
      return  (sum(model.p[(k, t)] for k in model.GENERATORS) 
               +  sum(model.b_discharge[(l, t)] for l in model.BATTERIES) 
               + model.s_minus[t] ==  model.s_plus[t] + model.demand[t]  
               + sum(model.b_charge[(l, t)] for l in model.BATTERIES))
    model.balance_rule = pyo.Constraint(model.HTIME, rule = balance_rule)

    # Defines operative cost
    def cost_op_rule(model,k, t):
      gen = generators_dict[k]
      return model.operative_cost[k, t] == (gen.f0 * gen.DG_max  
                                           * model.v[k, t] + gen.f1 *  model.p[k, t]) * model.fuel_cost
    model.cost_op_rule = pyo.Constraint(model.GENERATORS_DIESEL,
                                    model.HTIME, rule = cost_op_rule)
    
    # Batteries management
    def soc_rule(model, l, t):
      battery = batteries_dict[l]
      if t == 0:
          expr =  battery.eb_zero * (1 - battery.alpha)
          expr += model.b_charge[l, t] * battery.efc
          expr -= (model.b_discharge[l, t] / battery.efd)
          return model.soc[l, t] == expr 
      else:
          expr = model.soc[l, t - 1] * (1 - battery.alpha)
          expr += model.b_charge[l, t] * battery.efc
          expr -= (model.b_discharge[l, t] / battery.efd)
          return model.soc[l, t] == expr
    model.soc_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule = soc_rule)

    # Highest SOC level
    def Bconstraint_rule(model, l, t):
        battery = batteries_dict[l]
        return model.soc[l, t] <= battery.soc_max 
    model.Bconstraint = pyo.Constraint(model.BATTERIES, 
                                       model.HTIME, rule = Bconstraint_rule)
    
    # Minimum SOC level
    def Bconstraint2_rule(model, l, t):
      battery = batteries_dict[l]
      return model.soc[l, t] >= battery.soc_min 
    model.Bconstraint2 = pyo.Constraint(model.BATTERIES, 
                                        model.HTIME, rule = Bconstraint2_rule)
    
    # Maixmum level of energy that can enter to the battery
    def Bconstraint3_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_charge[l, t] <= battery.soc_max * model.bc[l, t] 
    model.Bconstraint3 = pyo.Constraint(model.BATTERIES, 
                                        model.HTIME, rule = Bconstraint3_rule)

    # Maximum level of energy that the battery can give to the microgrid
    def Bconstraint4_rule(model, l, t):
      battery = batteries_dict[l]
      return model.b_discharge[l, t] <= battery.soc_max * model.bd[l, t] 
    model.Bconstraint4 = pyo.Constraint(model.BATTERIES, 
                                        model.HTIME, rule = Bconstraint4_rule)
    
    # Charge and discharge control 
    def bcbd_rule(model, l, t):
      return  model.bc[l, t] + model.bd[l, t] <= 1
    model.bcbd_rule = pyo.Constraint(model.BATTERIES, model.HTIME, rule = bcbd_rule)
   
    # Defines operational rule relation Renewable - Diesel
    def op_relation_rule(model, t):
        check_tec = 0
        for i in generators_dict.values():
            if (i.tec == 'S' or i.tec == 'W'):
                check_tec = 1
                break
        if (check_tec == 1):
            expr4 = sum(model.demand[t1] for t1 in model.HTIME)
            return sum( model.p[k, t] for k in model.GENERATORS_REN) <= expr4 * (sum( model.v[k2,t] for k2 
                                                                         in model.GENERATORS_DIESEL)  + sum(model.soc[l, t] for l in model.BATTERIES)) 
        else:
            return pyo.Constraint.Skip
    model.op_relation_rule = pyo.Constraint(model.HTIME, rule = op_relation_rule)

    # Defines LPSP constraint
    def lpsp_cons_rule(model, t):
      if t >= (model.tlpsp - 1):
        rev_mov_average = sum(model.demand[t] for t in range((t - model.tlpsp + 1), t + 1)) 
        if rev_mov_average > 0:
          return (sum(model.s_minus[t] for t in range((t - model.tlpsp + 1), t + 1)) 
                  / rev_mov_average)  <= model.nse 
        else:
          return pyo.Constraint.Skip
      else:
        return pyo.Constraint.Skip
    model.lpsp_cons = pyo.Constraint(model.HTIME, rule = lpsp_cons_rule)
    
    #control that s- is lower than demand if tlpsp is two or more
    def additional_rule(model, t):
        if (model.tlpsp > 1):
            return model.s_minus[t] <= model.demand[t]
        else:
            return pyo.Constraint.Skip
    model.additional_rule = pyo.Constraint(model.HTIME, rule = additional_rule)

    #Constraints s- cost piecewise
    def sminusx_rule(model, t):
        return model.s_minus[t] == model.x1[t] + model.x2[t] + model.x3[t] + model.x4[t]
    model.sminusx_rule = pyo.Constraint(model.HTIME, rule = sminusx_rule)
    
    def sminusx1_rule(model, t):
        return model.x1[t] <= model.x1_limit * model.demand[t]
    model.sminusx1_rule = pyo.Constraint(model.HTIME, rule = sminusx1_rule)
    
    def sminusx2_rule(model, t):
        return model.x1[t] >= model.x1_limit * model.demand[t] * model.z1[t]
    model.sminusx2_rule = pyo.Constraint(model.HTIME, rule = sminusx2_rule)    

    def sminusx3_rule(model, t):
        return model.x2[t] <= (model.x2_limit - model.x1_limit) * model.demand[t] * model.z1[t]
    model.sminusx3_rule = pyo.Constraint(model.HTIME, rule = sminusx3_rule)
    
    def sminusx4_rule(model, t):
        return model.x2[t] >= (model.x2_limit - model.x1_limit) * model.demand[t] * model.z2[t]
    model.sminusx4_rule = pyo.Constraint(model.HTIME, rule = sminusx4_rule)  
    
    def sminusx5_rule(model, t):
        return model.x3[t] <= (model.x3_limit - model.x2_limit) * model.demand[t] * model.z2[t]
    model.sminusx5_rule = pyo.Constraint(model.HTIME, rule = sminusx5_rule)
    
    def sminusx6_rule(model, t):
        return model.x3[t] >= (model.x3_limit - model.x2_limit) * model.demand[t] * model.z3[t]
    model.sminusx6_rule = pyo.Constraint(model.HTIME, rule = sminusx6_rule)  

    def sminusx7_rule(model, t):
        return model.x4[t] <= (model.x4_limit - model.x3_limit) * model.demand[t] * model.z3[t]
    model.sminusx7_rule = pyo.Constraint(model.HTIME, rule = sminusx7_rule)    

    # Defines Objective function
    def obj_rule(model):
        tnpc_op = sum(sum(model.operative_cost[k, t] 
                          for t in model.HTIME) for k in model.GENERATORS_DIESEL)
        
        tnpc_op += sum(generators_dict[k].cost_rule 
                       for k in model.GENERATORS_REN)
        
        tnpc_op += sum(sum(model.b_discharge[l, t] 
                           * batteries_dict[l].cost_vopm for l in model.BATTERIES) for t in model.HTIME)
        
        lcoe_cost = model.tnpccrf + tnpc_op
        lcoe_cost +=  model.splus_cost * sum( model.s_plus[t] for t in model.HTIME)
        lcoe_cost += sum(model.x1_cost * model.x1[t] for t in model.HTIME)
        lcoe_cost += sum(model.x2_cost * model.x2[t] for t in model.HTIME)
        lcoe_cost += sum(model.x3_cost * model.x3[t] for t in model.HTIME)
        lcoe_cost += sum(model.x4_cost * model.x4[t] for t in model.HTIME)
        lcoe = lcoe_cost / (sum( model.demand[t] for t in model.HTIME))
        return lcoe
    model.LCOE_value = pyo.Objective(sense = pyo.minimize, rule = obj_rule)

    return model


def solve_model(model,
                Solver_data = {"MIP_GAP":0.01,"TEE_SOLVER":True,"OPT_SOLVER":"gurobi"}):
    '''
    Solve the pyomo model with specific optimizer

    Parameters
    ----------
    model 
    Solver_data : 
        MIP_GAP: GAP accepted for optimality
        TEE_SOLVER: display optimization status on screen
        OPT_SOLVER: Optimizer solver, example (Gurobi, CPLEX)

    Returns
    -------
    results 
    term : RESULTS DICTIONARY

    '''
    optimizer = Solver_data["OPT_SOLVER"]
    mipgap = Solver_data["MIP_GAP"]
    tee = Solver_data["TEE_SOLVER"]
    solver = pyo.SolverFactory(optimizer)
    solver.options['MIPGap'] = mipgap
    timea = time.time() #initial time
    results = solver.solve(model, tee = tee)
    term_cond = results.solver.termination_condition #termination condition
    #s_status = results.solver.status
    term = {}
    # http://www.pyomo.org/blog/2015/1/8/accessing-solver
    #status {aborted, unknown},
    #Termination Condition {maxTimeLimit (put), 
    #locallyOptimal, maxEvaluations, licesingProblem}
    
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
    def __init__(self, model, generators_dict, batteries_dict, stage):
        
        '''
        Load the results to the object associated to the class results

        Parameters
        ----------
        model
        generators_dictionary
        batteries_dictionary
        stage : one or two stage approach

        '''
        
        # Hourly data frame
        demand = pd.DataFrame(model.demand.values(), columns = ['demand'])
        
        #Generator data frame
        generation = {k: [0] * len(model.HTIME) for k in model.GENERATORS}
        for (k, t), f in model.p.items():
          generation [k][t] = value(f)
          
        generation = pd.DataFrame(generation, columns = [*generation.keys()])
               
        #Operative cost data frame
        generation_cost_data = {k + '_cost' : [0] * len(model.HTIME) for k in model.GENERATORS}
        #diesel cost
        for (k, t), f in model.operative_cost.items():
          generation_cost_data [k + '_cost'][t] = value(f)
        #renewable cost
        for k in model.GENERATORS:
            for t in model.HTIME:
                if generators_dict[k].tec != 'D':
                    generation_cost_data [k + '_cost'][t] = (generators_dict[k].gen_rule[t]
                                                             * generators_dict[k].cost_vopm) 
                
        generation_cost = pd.DataFrame(generation_cost_data, 
                                       columns = [*generation_cost_data.keys()])
        
        #Operative cost batteries
        batteries_cost_data = {l + '_cost' : [0] * len(model.HTIME) for l in model.BATTERIES}
        #Baterries cost
        for (l, t), f in model.b_discharge.items():
          batteries_cost_data [l + '_cost'][t] = value(f) * batteries_dict[l].cost_vopm
        
        batteries_cost = pd.DataFrame(batteries_cost_data, 
                                      columns = [*batteries_cost_data.keys()])
        
        # batery charge and discharge
        b_discharge_data = {l + '_b-' :[0] * len(model.HTIME) for l in model.BATTERIES}
        for (l, t), f in model.b_discharge.items():
          b_discharge_data [l + '_b-'][t] = value(f)
        
        b_discharge_df = pd.DataFrame(b_discharge_data, 
                                      columns = [*b_discharge_data.keys()])
        
        b_charge_data = {l +'_b+': [0] * len(model.HTIME) for l in model.BATTERIES}
        for (l, t), f in model.b_charge.items():
          b_charge_data [l + '_b+'][t] = value(f)
        b_charge_df = pd.DataFrame(b_charge_data, columns = [*b_charge_data.keys()])
        
        #SOC Battery dataframe
        soc_data = {l + '_soc' : [0] * len(model.HTIME) for l in model.BATTERIES}
        for (l, t), f in model.soc.items():
          soc_data [l + '_soc'][t] = value(f)
        soc_df = pd.DataFrame(soc_data, columns = [*soc_data.keys()])  
        
        # No supplied demand
        sminus_data = [0] * len(model.HTIME)
        lpsp_data = [0] * len(model.HTIME)
        for t in model.HTIME:
            sminus_data[t] = value(model.s_minus[t])
            if model.demand[t] != 0:
              lpsp_data [t] = value(model.s_minus[t]) / value(model.demand[t])
        
        sminus_df = pd.DataFrame(list(zip(sminus_data, lpsp_data)), 
                                 columns = ['S-','LPSP'])

        # Wasted Energy
        splus_data = [0] * len(model.HTIME)
        for t in model.HTIME:
            splus_data[t] = value(model.s_plus[t])
            
        splus_df = pd.DataFrame(splus_data, columns = ['Wasted Energy'])
        

        self.df_results = pd.concat([demand, generation, b_discharge_df, 
                                     b_charge_df, soc_df, sminus_df, splus_df, 
                                     generation_cost, batteries_cost], axis = 1) 
        
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
                if stage == 'One-Stage':
                    if generation[k].sum() > 0:
                        generators[k] = 1
                elif stage == 'Two-Stage':
                    generators[k] = 1
            self.descriptive['generators'] = generators
        # technologies
        tecno_data = {}
        try:
            for i in model.TECHNOLOGIES:
               tecno_data[i] = value(model.y[i])
            self.descriptive['technologies'] = tecno_data
        except: 
              pass 
              
        bat_data = {}
        try: 
            for l in model.BATTERIES:
               bat_data[l] = value(model.q[l])
            self.descriptive['batteries'] = bat_data
        except:
            for l in model.BATTERIES:
                if stage == 'One-Stage':
                    if b_discharge_df[l + '_b-'].sum() + b_charge_df[l + '_b+'].sum() > 0:
                        bat_data[l] = 1
                elif stage == 'Two-Stage':
                    bat_data[l] = 1
            self.descriptive['batteries'] = bat_data 
                  
        brand_data = {}
        try:
            for (i, j) in model.TEC_BRAND:
                brand_data[i, j] = value(model.x[i, j])  
            self.descriptive['Brand'] = brand_data
        except:
            pass
        #auxiliar area control
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
                
        bars.append(go.Bar(name ='Unsupplied Demand',x = df_results.index, 
                           y = df_results['S-']))
                
        plot = go.Figure(data=bars)
        
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
                              base = -1 * df_results['b+'],
                              marker_color = 'grey',
                              name = 'Charge'
                              ))
        
        # Change the bar mode
        plot.update_layout(barmode = 'stack')
        
        
        return plot

