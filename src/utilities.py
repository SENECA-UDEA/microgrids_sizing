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
import copy



def read_data(demand_filepath, 
              forecast_filepath,
              units_filepath,
              instance_data):
    
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
    
    try:
        instance_data =  requests.get(instance_data)
        instance_data = json.loads(instance_data.text)
    except:
        f = open(instance_data)
        instance_data = json.load(f) 
    
    return demand_df, forecast_df, generators, batteries, instance_data


def create_objects(generators, batteries, forecast_df, demand_df, instance_data):
    # Create generators and batteries
    generators_dict = {}
    for k in generators:
      if k['tec'] == 'S':
        obj_aux = Solar(*k.values())
        obj_aux.Get_INOCT(instance_data["caso"], instance_data["w"])
        obj_aux.Solargeneration( forecast_df['t_ambt'], forecast_df['DNI'], instance_data["G_stc"])
      elif k['tec'] == 'W':
        obj_aux = Eolic(*k.values())
        obj_aux.Windgeneration(forecast_df['Wt'],instance_data["h2"],instance_data["coef_hel"] )
      elif k['tec'] == 'D':
        obj_aux = Diesel(*k.values())   
      generators_dict[k['id_gen']] = obj_aux
      
    batteries_dict = {}
    for l in batteries:
        obj_aux = Battery(*l.values())
        batteries_dict[l['id_bat']] = obj_aux
        batteries_dict[l['id_bat']].calculatesoc()
    return generators_dict, batteries_dict


def create_technologies(generators_dict, batteries_dict):
    # Create technologies dictionarry
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
    renewables_dict = dict()
    for gen in generators_dict.values(): 
        if gen.tec == 'S' or gen.tec == 'W': #or gen.tec = 'H'
          if not (gen.tec in renewables_dict.keys()):
              renewables_dict[gen.tec] = set()
              renewables_dict[gen.tec].add(gen.br)
          else:
              renewables_dict[gen.tec].add(gen.br)
              
    return technologies_dict, renewables_dict
 
    
#calculate total cost for two stage approach
def calculate_sizingcost(generators_dict, batteries_dict, ir, years):
            expr = 0
            for gen in generators_dict.values(): 
                expr += gen.cost_up*gen.n 
                expr += gen.cost_r*gen.n 
                expr += gen.cost_fopm*gen.n 
                expr -= gen.cost_s*gen.n 
                
            for bat in batteries_dict.values(): 
                expr += bat.cost_up
                expr += bat.cost_fopm
                expr += bat.cost_r
                expr -= bat.cost_s
                
                
                
            TNPC = expr
            CRF = (ir * (1 + ir)**(years))/((1 + ir)**(years)-1) 

            return TNPC, CRF


def calculate_area (sol_actual):
    solution = copy.deepcopy(sol_actual)
    dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
    area = 0
    for i in dict_actual.values():
        area += i.area 
    return area


def script_generators(generators, amax):
    aux_generators = []
    generators_def = []
    generators_transformed = copy.deepcopy(generators)
    
    for i in generators_transformed:
        if (i['tec'] == 'S' or i['tec'] == 'W'):
            #calculate the number of n that can create
            total_n = math.floor(amax/i['area'])
            #save the name of the generator
            name = i['id_gen']
            #save the area
            area = i['area']
            #cont to rename the new object of the dictionary
            cont = 1
            for j in range(1, total_n + 1):
                aux_generators = []
                aux_generators = i
                #create a name
                aux_generators['id_gen'] = name + ' n: ' + str(cont)
                cont = cont + 1
                #change the n
                aux_generators['n'] = j
                aux_generators['area'] = j * area
                generators_def.append(copy.deepcopy(aux_generators))
        else:
            #Diesel
            name = i['id_gen']
            aux_generators = []
            aux_generators = i
            aux_generators['id_gen'] = name
            aux_generators['n'] = 1
            generators_def.append(copy.deepcopy(aux_generators))
            
    '''        
    with open("student.json", "w") as write_file:
        json.dump(generators_def, write_file, indent=4)
    '''
    return generators_def

def calculate_energy(batteries_dict, generators_dict, model_results, demand_df):  
   column_data = {}
   energy_data = {}
   aux_energy_data = []
   renew_data = {}
   aux_renew_data = []
   total_data = [0]*len(demand_df)
   aux_total_data = []
   brand_data = {}
   aux_brand_data = []
   for bat in batteries_dict.values(): 
       if (model_results.descriptive['batteries'][bat.id_bat] == 1):
           column_data[bat.id_bat+'_%'] =  model_results.df_results[bat.id_bat+'_b-'] / model_results.df_results['demand']
           aux_total_data = model_results.df_results[bat.id_bat+'_b-']
           total_data += aux_total_data
           key_energy_total = bat.tec+ 'total'
           key_brand_total = bat.br + 'total'
           if key_energy_total in energy_data:
               aux_energy_data = []
               aux_energy_data = energy_data[key_energy_total] +  model_results.df_results[bat.id_bat+'_b-']
               energy_data[key_energy_total] = aux_energy_data
           else:
               energy_data[key_energy_total] =  model_results.df_results[bat.id_bat+'_b-']           
    
           if key_brand_total in brand_data:
               aux_brand_data = []
               aux_brand_data = brand_data[key_brand_total] +  model_results.df_results[bat.id_bat+'_b-']
               brand_data[key_brand_total] = aux_brand_data
           else:
               brand_data[key_brand_total] =  model_results.df_results[bat.id_bat+'_b-']           
      
   for gen in generators_dict.values():
       if (model_results.descriptive['generators'][gen.id_gen] == 1):
           column_data[gen.id_gen+'_%'] =  model_results.df_results[gen.id_gen] / model_results.df_results['demand']
           key_energy_total = gen.tec + 'total'
           key_renew_total = gen.tec + 'total'
           key_brand_total = gen.br + 'total'
           total_data += model_results.df_results[gen.id_gen]
           if key_energy_total in energy_data:
               aux_energy_data = []
               aux_energy_data = energy_data[key_energy_total] +  model_results.df_results[gen.id_gen]
               energy_data[key_energy_total] = aux_energy_data
           else:
               energy_data[key_energy_total] =  model_results.df_results[gen.id_gen]           
           
           if key_brand_total in brand_data:
               aux_brand_data = []
               aux_brand_data = brand_data[key_brand_total] +  model_results.df_results[gen.id_gen]
               brand_data[key_brand_total] = aux_brand_data
           else:
               brand_data[key_brand_total] =  model_results.df_results[gen.id_gen]           
           if (gen.tec == 'S' or gen.tec == 'W'):
               if key_renew_total in renew_data:
                   aux_renew_data = []
                   aux_renew_data = renew_data[key_renew_total] +  model_results.df_results[gen.id_gen]
                   renew_data[key_renew_total] = aux_renew_data
               else:
                   renew_data[key_renew_total] =  model_results.df_results[gen.id_gen]           

   percent_df = pd.DataFrame(column_data, columns=[*column_data.keys()])
   energy_df = pd.DataFrame(energy_data, columns=[*energy_data.keys()])
   renew_df = pd.DataFrame(renew_data, columns=[*renew_data.keys()])
   arraydf = np.array(total_data)
   total_df = pd.DataFrame(arraydf, columns=['Total energy'])
   brand_df = pd.DataFrame(brand_data, columns=[*brand_data.keys()])
   
   return percent_df, energy_df, renew_df, total_df, brand_df


'''

def min2hms(hm):
    """conversion min -> (hours, min, sec)
    
    """
    H = int(hm/60)
    M = ((hm/60)-H)*60
    m = int(M)
    S = (M-m)*60
    s = int(S)
    return H,m,s

def hd2hms(hd):
    """decimal time conversion ->
       (hours,minutes,seconds)
    """
    H = int(hd)
    m = (hd - H)*60
    M = int(m)
    s = m-M
    S = int(s*60)
    return H,M,S

def Get_SolarP01(LT,TZ,dia,Long,Latit):
    """LT: local time(hour)
       TZ: time zone
       dia: counted from January 1
       Long: longitude in degrees
       Latit: latitude in degrees
       sun position calculation
       return: Elevation,Azimuth in degrees
       version: 2019-02-05
       ref:https://www.pveducation.org/pvcdrom/2-properties-sunlight/suns-position
    """
    ka = 180/np.pi
    LSTM = 15*(TZ)#Local Standard Time Meridian(LSTM)
    EoT = lambda x:9.87*np.sin(2*x)-7.53*np.cos(x)-1.5*np.sin(x)#x in radians
    B = lambda d:((360/365)*d - 81)*(np.pi/180)
    LT1 = LT*60 #conversion to minutes
    TC = 4*(Long - LSTM) + EoT(B(dia))#Time Correction Factor (TC)
    LST = LT1 + (TC/60)#The Local Solar Time (LST)
    HRA = 15*((LST/60)-12)#Hour Angle (HRA)
    delta = 23.45*np.sin(B(dia))#declination angle (delta)
    Elevation = np.arcsin(np.sin(delta*np.pi/180)*np.sin(Latit*np.pi/180)+np.cos(delta*np.pi/180)*np.cos(Latit*np.pi/180)*np.cos(HRA*np.pi/180))
    
    ##calculate Azimuth 
    ## asumes teta:latitude
    k_num = np.sin(delta*np.pi/180)*np.cos(Latit*np.pi/180)+np.cos(delta*np.pi/180)*np.sin(Latit*np.pi/180)*np.cos(HRA*np.pi/180)
    k_total = k_num/np.cos(Elevation)

    if abs(k_total)>1.0:## vancouver essay
        k_total = k_total/abs(k_total)
    # print(k_total)

    Azimuth = np.arccos(k_total)
    if min2hms(LST)[0]>=12:#Correction after noon
        Azimuth = 2*np.pi - Azimuth

    return Elevation*ka, Azimuth*ka 
            #a_s: Sun altitude (grados)//Elevation
            #A_s: Sun Azimuth (grados)

def cos_AOI(a_M,A_M,a_s,A_s):
    """AOI: angle of incidence
        a_M: Module altitude (grados)
        A_M: Module Azimuth (grados)
        a_s: Sun altitude (grados)
        A_s: Sun Azimuth (grados)
        """
    Ar = A_M - A_s
    c1 = np.cos(a_M*np.pi/180)*np.cos(a_s*np.pi/180)*np.cos(Ar*np.pi/180)
    c2 = np.sin(a_M*np.pi/180)*np.sin(a_s*np.pi/180)
    ct = c1 + c2
    return ct

def Gdirect(SF,DNI,cosAOI):
    """Gdirect: direct Irradiance on module
        SF: shading factor
        DNI: direct normal irradiance
        cosAOI: cos angle of incidence
    """
    Gd = SF*DNI*cosAOI
    if Gd < 0:
        Gd = 0 #negative Direct Irradiance on the PV module as zero
        
    return Gd

def Get_SVF01(t_M):
    """t_M: tilted angle of Module (grados)
        Free Horizont model
    """
    svf = (1 + np.cos(t_M*np.pi/180))/2
    return svf

def Gdiffuse(SVF,DHI):
    """Gdiffuse: diffuse Irradiance on module
        SVF: sky view factor
        DHI: diffuse horizontal irradiance
        Isotropic sky model
    """
    Gd = SVF*DHI
    return Gd

def Galbedo(a_gnd,SVF,GHI):
    """Galbedo: ground irradiance
        a_gnd: albedo coefficient
        SVF: Sky View Factor
        GHI: Global Horizontal Irradiance
    """
    Ga = a_gnd*(1 - SVF)*GHI
    return Ga


def Gmodule(G_dr,G_df,G_alb):
    """Gmodule: Total Irradiance
        G_dr: direct
        G_df: diffuse
        G_alb: ground
    """
    Gm = G_dr + G_df + G_alb
    return Gm  

    
def irradiance_panel (forecast_df, instance_data):
    theta_M = instance_data["tilted_angle"]
    a_M = 90 - theta_M
    A_M = instance_data["module_azimuth"]
    TZ = instance_data["time_zone"]
    long = instance_data["longitude"]
    latit = instance_data["latitude"]
    alpha = instance_data["alpha_albedo"]
    caso = instance_data["caso"] #Direct Mount, Stand off or Rack Mount
    w = instance_data["w"] #distance to the mount
    t_amb = forecast_df['t_ambt'] #Room temperature
    gt_data = {}
    for t in list(forecast_df['t'].index.values):
        LT = forecast_df['t'][t]
        SF1 = forecast_df['SF'][t] #Shadow factor
        DNI = forecast_df['DNI'][t] #Direct normal Irradiance
        DHI = forecast_df['DHI'][t] #Diffuse Horizontal Irradiance
        GHI = forecast_df['GHI'][t] #Global horizontal Irradiance
        day = forecast_df['day'][t] #Day of the year
        
        Gs = Get_SolarP01(LT,TZ,day,long,latit) #Sum altitude and sum Azimuth   
        ds = cos_AOI(a_M,A_M,Gs[0],Gs[1]) #COsine incidence angle
        gdr = Gdirect(SF1,DNI,ds) #Direct irradiance
        svf = Get_SVF01(theta_M) #Sky view factor
        gdf = Gdiffuse(svf,DHI) #Diffuse irradiance
        galb = Galbedo(alpha,svf,GHI) #Groud irradiance
        gt_data[t] = Gmodule(gdr, gdf, galb) #Total irradiance
    
    gt =  pd.DataFrame(gt_data.items(), columns = ['t','gt']) 
        
    return  w, caso, t_amb, gt
'''
