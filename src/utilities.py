# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:51:07 2022

@author: pmayaduque
"""
from classes import Solar, Eolic, Diesel, Battery
import pandas as pd
import requests
import json 
import numpy as np
import math


def read_data(demand_filepath, 
              forecast_filepath,
              units_filepath):
    
    forecast_df = pd.read_csv(forecast_filepath)
    demand_df = pd.read_csv(demand_filepath)
    try:
        generators_data =  requests.get(units_filepath)
        generators_data = json.loads(generators_data.text)
    except:
        f = open(units_filepath)
        generators_data = json.load(f)
    
    generators = generators_data['generators']
    batteries = generators_data['batteries']
    
    return demand_df, forecast_df, generators, batteries

def create_objects(generators, batteries):
    # Create generators and batteries
    generators_dict = {}
    for k in generators:
      if k['tec'] == 'S':
        obj_aux = Solar(*k.values())
      elif k['tec'] == 'W':
        obj_aux = Eolic(*k.values())
      elif k['tec'] == 'D':
        obj_aux = Diesel(*k.values())      
      generators_dict[k['id_gen']] = obj_aux
      
    batteries_dict = {}
    for l in batteries:
        obj_aux = Battery(*l.values())
        batteries_dict[l['id_bat']] = obj_aux
        batteries_dict[l['id_bat']].soc_dod()
        
    # Create technologies list
    technologies_dict = dict()
    for bat in batteries_dict.values(): 
      if not (bat.tec in technologies_dict.keys()):
        technologies_dict[bat.tec] = set()
        technologies_dict[bat.tec].add(bat.br)
      else:
        technologies_dict[bat.tec].add(bat.br)
    for gen in generators_dict.values(): 
      if not (gen.tec in technologies_dict.keys()):
        technologies_dict[gen.tec] = set()
        technologies_dict[gen.tec].add(gen.br)
      else:
        technologies_dict[gen.tec].add(gen.br)

    # Creates renewables dict
    #another attribute could be created to the class, for the user to determine if it is renewable or not
    renewables_dict = dict()
    for gen in generators_dict.values(): 
        if gen.tec == 'S' or gen.tec == 'W': #or gen.tec = 'H'
          if not (gen.tec in renewables_dict.keys()):
              renewables_dict[gen.tec] = set()
              renewables_dict[gen.tec].add(gen.br)
          else:
              renewables_dict[gen.tec].add(gen.br)
              
    return generators_dict, batteries_dict, technologies_dict, renewables_dict

def generation(gen, t, forecast_df):

      if gen.tec == 'S':
         g_rule = gen.ef * gen.G_test * (forecast_df['Rt'][t]/gen.R_test) 
      elif gen.tec == 'W':
          if forecast_df['Wt'][t] < gen.w_min:
              g_rule = 0
          elif forecast_df['Wt'][t] < gen.w_a:
              g_rule =  ((1/2) * gen.p * gen.s * (forecast_df['Wt'][t]**3) * gen.ef * gen.n )/1000
              #p en otros papers es densidad del aíre, preguntar a Mateo, por qué divide por 1000?
          elif forecast_df['Wt'][t] <= gen.w_max:
              g_rule = ((1/2) * gen.p * gen.s * (gen.w_a**3) * gen.ef * gen.n )/1000
          else:
              g_rule = 0
      elif gen.tec == 'D':
         g_rule =  gen.c_max
      return g_rule
  
    
def Calculate_Infraes_cost(generators_opt, batteries_opt):
     TNPC = sum(generators_opt[k].cost_up for k in generators_opt) + sum(batteries_opt[l].cost_up for l in batteries_opt) 
     TNPC += sum(generators_opt[k].cost_r for k in generators_opt) + sum(batteries_opt[l].cost_r for l in batteries_opt) 
     TNPC += sum(generators_opt[k].cost_om for k in generators_opt) + sum(batteries_opt[l].cost_om for l in batteries_opt) 
     TNPC -= sum(generators_opt[k].cost_s for k in generators_opt) + sum(batteries_opt[l].cost_s for l in batteries_opt) 
                
     return TNPC