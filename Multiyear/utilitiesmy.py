# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:51:07 2022

@author: scastellanos
"""
from multiyear.classesmy import Solar, Eolic, Diesel, Battery
import pandas as pd
import requests
import json 
import numpy as np
import copy
import scipy.stats as st
import math

def read_data(demand_filepath, 
              forecast_filepath,
              units_filepath,
              instance_filepath,
              fiscal_filepath,
              cost_filepath,
              my_filepath):
    
    forecast_df = pd.read_csv(forecast_filepath)
    demand_df = pd.read_csv(demand_filepath)
    try:
        generators_data =  requests.get(units_filepath)
        generators_data = json.loads(generators_data.text)
    except:
        f = open(units_filepath)
        generators_data = json.load(f)    

    try:
        generators = generators_data['generators']
    except:
        generators = {}

    try:
        batteries = generators_data['batteries']
    except: 
        batteries = {}
        
    try:
        instance_data = requests.get(instance_filepath)
        instance_data = json.loads(instance_data.text)
    except:
        f = open(instance_filepath)
        instance_data = json.load(f) 

    try:
        fiscal_data = requests.get(fiscal_filepath)
        fiscal_data = json.loads(fiscal_data.text)
    except:
        f = open(fiscal_filepath)
        fiscal_data = json.load(f) 

    try:
        cost_data = requests.get(cost_filepath)
        cost_data = json.loads(cost_data.text)
    except:
        f = open(cost_filepath)
        cost_data = json.load(f) 
        
    try:
        my_data = requests.get(my_filepath)
        my_data = json.loads(my_data.text)
    except:
        f = open(my_filepath)
        my_data = json.load(f) 
        
    return demand_df, forecast_df, generators, batteries, instance_data, fiscal_data, cost_data, my_data


def create_objects(generators, batteries, forecast_df, 
                   demand_df, instance_data, my_data):
    
    '''Create generators and batteries'''
    generators_dict = {}
    for k in generators:
      if k['tec'] == 'S':
        obj_aux = Solar(*k.values())
        irr = irradiance_panel (forecast_df, instance_data)
        obj_aux.get_inoct(instance_data["caso"], instance_data["w"])
        obj_aux.solar_generation( forecast_df['t_ambt'], irr,
                                 instance_data["G_stc"], my_data["sol_deg"])
        obj_aux.solar_cost()
      elif k['tec'] == 'W':
        obj_aux = Eolic(*k.values())
        obj_aux.eolic_generation(forecast_df['Wt'], instance_data["h2"]
                                 , instance_data["coef_hel"], my_data["wind_deg"] )
        obj_aux.eolic_cost()
      elif k['tec'] == 'D':
        obj_aux = Diesel(*k.values())   
        
      generators_dict[k['id_gen']] = obj_aux
      
    batteries_dict = {}
    for l in batteries:
        obj_aux = Battery(*l.values())
        batteries_dict[l['id_bat']] = obj_aux
        batteries_dict[l['id_bat']].calculate_soc()
    return generators_dict, batteries_dict


def create_technologies(generators_dict, batteries_dict):
    '''Create technologies dictionary'''
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

    

def calculate_inverter_cost(generators_dict, batteries_dict, inverter_cost):
            '''calculate inverter cost with the instaled generators and batteries'''
            expr = 0
            
            for gen in generators_dict.values(): 
                if (gen.tec == 'D'): 
                    #cost as percentage of rated size network
                    expr += gen.DG_max * inverter_cost
                elif (gen.tec == 'S'):
                    #cost as percentage of rated size network
                    expr += gen.Ppv_stc * inverter_cost                    
                else:
                    #cost as percentage of rated size network
                    expr += gen.P_y * inverter_cost

            for bat in batteries_dict.values(): 
                #cost as percentage of rated size network
                expr += bat.soc_max * inverter_cost

            return expr


#calculate total cost for two stage approach
def calculate_sizing_cost(generators_dict, batteries_dict,
                          ir, years, delta, inverter):
    
            expr = 0
            
            for gen in generators_dict.values(): 
                if (gen.tec != 'D'): 
                    #fiscal incentive if not diesel
                    expr += gen.cost_up * delta 
                    expr += gen.cost_r * delta
                else:
                    expr += gen.cost_up
                    expr += gen.cost_r 

                expr -= gen.cost_s 
                expr += gen.cost_fopm 

            for bat in batteries_dict.values(): 
                expr += bat.cost_up * delta
                expr += bat.cost_r * delta
                expr -= bat.cost_s
                expr += bat.cost_fopm
    
            #Operative cost doesn't take into account the crf
            tnpccrf = (expr + inverter) * (1 + ir)
    
            return tnpccrf


def calculate_area (sol_actual):
    solution = copy.deepcopy(sol_actual)
    dict_actual = {**solution.generators_dict_sol, **solution.batteries_dict_sol}
    area = 0
    for i in dict_actual.values():
        area += i.area 

    return area



def calculate_energy(batteries_dict, generators_dict, model_results, demand_df):  
   '''Calculate energy total, for eevery brand, technology or renewable '''
   column_data = {}
   energy_data = {}
   aux_energy_data = []
   renew_data = {}
   aux_renew_data = []
   total_data = [0] * len(demand_df)
   aux_total_data = []
   brand_data = {}
   aux_brand_data = []
   for bat in batteries_dict.values(): 
       #check that the battery is installed
       if (model_results.descriptive['batteries'][bat.id_bat] == 1):
           column_data[bat.id_bat + '_%'] =  model_results.df_results[bat.id_bat
                                                                    + '_b-'] / model_results.df_results['demand']
           
           aux_total_data = model_results.df_results[bat.id_bat + '_b-']
           #sum all generation
           total_data += aux_total_data
           #check the key for create or continue in the same dict
           key_energy_total = bat.tec + 'total'
           key_brand_total = bat.br + 'total'
           if key_energy_total in energy_data:
               aux_energy_data = []
               aux_energy_data = (energy_data[key_energy_total] 
                                  + model_results.df_results[bat.id_bat + '_b-'])
               #fill the dictionary
               energy_data[key_energy_total] = aux_energy_data
           else:
               energy_data[key_energy_total] = model_results.df_results[bat.id_bat + '_b-']           
    
           if key_brand_total in brand_data:
               aux_brand_data = []
               aux_brand_data = (brand_data[key_brand_total] 
                                 + model_results.df_results[bat.id_bat + '_b-'])
               
               #fill the dictionary
               brand_data[key_brand_total] = aux_brand_data
           else:
               brand_data[key_brand_total] =  model_results.df_results[bat.id_bat + '_b-']           
      
   for gen in generators_dict.values():
       #check that the generator is installed
       if (model_results.descriptive['generators'][gen.id_gen] == 1):
           column_data[gen.id_gen + '_%'] =  (model_results.df_results[gen.id_gen] 
                                             / model_results.df_results['demand'])
           
           #check the key for create or continue in the same dict
           key_energy_total = gen.tec + 'total'
           key_renew_total = gen.tec + 'total'
           key_brand_total = gen.br + 'total'
           #sum all generation
           total_data += model_results.df_results[gen.id_gen]
           if key_energy_total in energy_data:
               aux_energy_data = []
               aux_energy_data = (energy_data[key_energy_total] 
                                  + model_results.df_results[gen.id_gen])
               #fill the dictionary
               energy_data[key_energy_total] = aux_energy_data
           else:
               energy_data[key_energy_total] =  model_results.df_results[gen.id_gen]           
           
           if key_brand_total in brand_data:
               aux_brand_data = []
               aux_brand_data = (brand_data[key_brand_total]
                                 + model_results.df_results[gen.id_gen])
               #fill the dictionary
               brand_data[key_brand_total] = aux_brand_data
           else:
               brand_data[key_brand_total] =  model_results.df_results[gen.id_gen]           
           
           if (gen.tec == 'S' or gen.tec == 'W'):
               if key_renew_total in renew_data:
                   aux_renew_data = []
                   aux_renew_data = (renew_data[key_renew_total] 
                                     + model_results.df_results[gen.id_gen])
                   #fill the dictionary
                   renew_data[key_renew_total] = aux_renew_data
               else:
                   renew_data[key_renew_total] =  model_results.df_results[gen.id_gen]           

   #Create dataframes
   percent_df = pd.DataFrame(column_data, columns = [*column_data.keys()])
   energy_df = pd.DataFrame(energy_data, columns = [*energy_data.keys()])
   renew_df = pd.DataFrame(renew_data, columns = [*renew_data.keys()])
   arraydf = np.array(total_data)
   total_df = pd.DataFrame(arraydf, columns = ['Total energy'])
   brand_df = pd.DataFrame(brand_data, columns = [*brand_data.keys()])
   
   return percent_df, energy_df, renew_df, total_df, brand_df


def interest_rate (i_f, inf):
    #inf = inflation
    #i_f = nominal rate
    ir = (i_f - inf) / (1 + inf)
    return ir


def calculate_cost_data(generators, batteries, instance_data,
                        parameters_cost):
    #inflation
    inf = instance_data['inf']
    #nominal rate
    i_f = instance_data['i_f']
    years = instance_data['years']
    ir = interest_rate(i_f,inf)
    #defaul useful life Diesel and batteries = 10
    life_cicle = parameters_cost['life_cicle']
    n_cycles = years/life_cicle    
    #Calculate tax for remplacement
    tax = 0
    for h in range(1, int(n_cycles) + 1):
        tax += 1 / ((1 + ir)**(h * life_cicle))
        
    aux_generators = []
    aux_batteries = []
    #definitive list
    generators_def = []
    batteries_def = []
    #Calculate costs with investment cost
    for i in generators:
        if (i['tec'] == 'S'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = parameters_cost['param_r_solar']
            
            aux_generators['cost_s'] = cost_up * parameters_cost['param_s_solar'] * (((1 + inf)
                                                                                      /(1 + ir))**years)
            
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_solar'] 
            aux_generators['cost_vopm'] = parameters_cost['param_v_solar']      
        elif (i['tec'] == 'W'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = parameters_cost['param_r_wind']  
            aux_generators['cost_s'] = cost_up * parameters_cost['param_s_wind'] * (((1 + inf)
                                                                                     / (1 + ir))**years)
            
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_wind']  
            aux_generators['cost_vopm'] = parameters_cost['param_v_wind']  
            
        elif (i['tec'] == 'D'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = cost_up * parameters_cost['param_r_diesel'] * tax
            aux_generators['cost_s'] = cost_up * parameters_cost['param_s_diesel'] * (((1 + inf)
                                                                                      / (1 + ir))**years)
            
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_diesel']

        generators_def.append(copy.deepcopy(aux_generators))  
        
    for i in batteries:
        cost_up = i['cost_up']
        aux_batteries = []
        aux_batteries = i
        aux_batteries['cost_r'] = cost_up * parameters_cost['param_r_bat'] * tax
        aux_batteries['cost_s'] = cost_up * parameters_cost['param_s_bat'] * (((1 + inf)
                                                                               / (1 + ir))**years)
        aux_batteries['cost_fopm'] = cost_up * parameters_cost['param_f_bat']
        aux_batteries['cost_vopm'] = cost_up * parameters_cost['param_v_bat']
        batteries_def.append(copy.deepcopy(aux_batteries))

    return generators_def, batteries_def


def fiscal_incentive (credit, depreciation, corporate_tax, ir, T1, T2):
    '''
    corporate_tax = effective corporate tax income rate
    Credit = investment tax credit
    Depreciation = factor expressed as % of investment cost over T2 year
    ir = Interest rate
    T1 = Maximum number of years to apply the investment tax credit
    T2 = useful life of the power generating facility in year - depreciation
    '''
    delta = 0
    expr = 0
    for j in range(1, int(T1) + 1):
        expr += credit / ((1 + ir)**j)

    for j in range(1, int(T2) + 1):
        expr += depreciation / ((1 + ir)**j)
    
    delta = (1 / (1 - corporate_tax)) * (1 - corporate_tax * expr)
    
    return delta

    
def irradiance_panel (forecast_df, instance_data):
 
    if (forecast_df['GHI'].sum() <= 0 or forecast_df['DHI'].sum() <= 0):
        #Default only DNI if it is not GHI or DHI
        irr_data = forecast_df['DNI']
    else:       
        theta_M = instance_data["tilted_angle"]
        a_M = 90 - theta_M
        A_M = instance_data["module_azimuth"]
        TZ = instance_data["time_zone"]
        long = instance_data["longitude"]
        latit = instance_data["latitude"]
        alpha = instance_data["alpha_albedo"]
        SF1 = instance_data['shading factor']
        irr_data = {}
        for t in list(forecast_df['t'].index.values):
            LT = forecast_df['t'][t]
            DNI = forecast_df['DNI'][t] #Direct normal Irradiance
            DHI = forecast_df['DHI'][t] #Diffuse Horizontal Irradiance
            GHI = forecast_df['GHI'][t] #Global horizontal Irradiance
            day = forecast_df['day'][t] #Day of the year
            Gs = get_solar_parameters(LT, TZ, day, long, latit) #Sum altitude and sum Azimuth   
            ds = cos_incidence_angle(a_M, A_M, Gs[0], Gs[1]) #COsine incidence angle
            svf = get_sky_view_factor(theta_M) #Sky view factor
            G_dr = SF1 * DNI * ds
            if G_dr < 0:
                G_dr = 0 #negative Direct Irradiance on the PV module as zero

            G_df = svf * DHI #Diffuse irradiancia
            G_alb = alpha*(1 - svf) * GHI #Groud irradiance
            irr_data[t] = G_dr + G_df + G_alb #Total irradiance
    
    irr =  pd.DataFrame(irr_data.items(), columns = ['t','irr']) 
        
    return  irr


def min_to_hms(hm):
    """conversion min -> (hours, min, sec)
    """
    H = int(hm / 60)
    M = ((hm / 60) - H) * 60
    m = int(M)
    S = (M - m) * 60
    s = int(S)
    return H, m, s


def decimal_hour_to_hms(hd):
    """decimal time conversion ->
       (hours,minutes,seconds)
    """
    H = int(hd)
    m = (hd - H) * 60
    M = int(m)
    s = m - M
    S = int(s * 60)
    return H, M, S


def get_solar_parameters(LT, TZ, dia, Long, Latit):
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
    ka = 180 / np.pi
    LSTM = 15 * TZ#Local Standard Time Meridian(LSTM)
    EoT = lambda x:9.87 * np.sin(2 * x) - 7.53 * np.cos(x) - 1.5 * np.sin(x)#x in radians
    B = lambda d:((360 / 365) * d - 81)*(np.pi / 180)
    LT1 = LT * 60 #conversion to minutes
    TC = 4 * (Long - LSTM) + EoT(B(dia))#Time Correction Factor (TC)
    LST = LT1 + (TC / 60)#The Local Solar Time (LST)
    HRA = 15 * ((LST / 60) - 12)#Hour Angle (HRA)
    delta = 23.45 * np.sin(B(dia))#declination angle (delta)
    Elevation = (np.arcsin(np.sin(delta * np.pi / 180) * np.sin(Latit * np.pi / 180)
                           + np.cos(delta * np.pi / 180) * np.cos(Latit * np.pi / 180) * np.cos(HRA * np.pi / 180)))    
    ##calculate Azimuth 
    ## asumes teta:latitude
    k_num = (np.sin(delta * np.pi / 180)*np.cos(Latit * np.pi / 180) + np.cos(delta * np.pi / 180)
             *np.sin(Latit * np.pi / 180) * np.cos(HRA * np.pi / 180))
    k_total = k_num / np.cos(Elevation)

    if abs(k_total) > 1.0:## vancouver essay
        k_total = k_total / abs(k_total)
    # print(k_total)

    Azimuth = np.arccos(k_total)
    if min_to_hms(LST)[0] >= 12:#Correction after noon
        Azimuth = 2 * np.pi - Azimuth

    return Elevation * ka, Azimuth * ka 


#a_s: Sun altitude (grados)//Elevation
#A_s: Sun Azimuth (grados)
def cos_incidence_angle(a_M, A_M, a_s, A_s):
    """AOI: angle of incidence
        a_M: Module altitude (degrees)
        A_M: Module Azimuth (degrees)
        a_s: Sun altitude (degrees)
        A_s: Sun Azimuth (degrees)
        """
    Ar = A_M - A_s
    c1 = np.cos(a_M * np.pi / 180)*np.cos(a_s * np.pi / 180)*np.cos(Ar * np.pi / 180)
    c2 = np.sin(a_M * np.pi / 180)*np.sin(a_s * np.pi / 180)
    ct = c1 + c2
    return ct


def get_sky_view_factor(t_M):
    """t_M: tilted angle of Module (grados)
        Free Horizont model
    """
    svf = (1 + np.cos(t_M * np.pi / 180)) / 2
    return svf


def calculate_multiyear_data(demand_df, forecast_df, my_data, years):
    
    #total hours
    len_total = 8760 * years
    aux_demand = {k : [0] * (len_total) for k in demand_df}
    aux_forecast = {k : [0] * (len_total) for k in forecast_df}
    
    for i in range(len_total):
        aux_demand['t'][i] = i
        aux_forecast['t'][i] = i
        #first year same 
        if (i < 8760):    
            aux_demand['demand'][i]= demand_df['demand'][i]
            aux_forecast['DNI'][i] = forecast_df['DNI'][i]
            aux_forecast['t_ambt'][i] = forecast_df['t_ambt'][i]
            aux_forecast['Wt'][i] = forecast_df['Wt'][i]
            aux_forecast['Qt'][i] = forecast_df['Qt'][i]
            aux_forecast['GHI'][i] = forecast_df['GHI'][i]
            aux_forecast['day'][i] = forecast_df['day'][i]
            aux_forecast['SF'][i] = forecast_df['SF'][i]
            aux_forecast['DHI'][i] = forecast_df['DHI'][i]
        #others years
        else:
            #get the year
            k = math.floor(i / 8760)
            #apply tax
            val = demand_df['demand'][i - 8760 * k] * (1 + my_data["demand_tax"]) ** k
            #asign value
            aux_demand['demand'][i] = val
            #forecast is the same that first year
            val2 = forecast_df['DNI'][i - 8760 * k]
            aux_forecast['DNI'][i] = val2
            aux_forecast['t_ambt'][i] = forecast_df['t_ambt'][i - 8760 * k]
            val3 = forecast_df['Wt'][i - 8760 * k]
            aux_forecast['Wt'][i] = val3
            aux_forecast['Qt'][i] = forecast_df['Qt'][i - 8760 * k]
            val4 = forecast_df['GHI'][i - 8760 * k]
            aux_forecast['GHI'][i] = val4
            aux_forecast['day'][i] = forecast_df['day'][i - 8760 * k]
            aux_forecast['SF'][i] = forecast_df['SF'][i - 8760 * k]
            val5 = forecast_df['DHI'][i - 8760 * k]
            aux_forecast['DHI'][i] = val5      
            
    #create dataframe
    demand = pd.DataFrame(aux_demand, columns=['t','demand'])
    forecast = pd.DataFrame(aux_forecast, columns=['t','DNI','t_ambt','Wt', 
                                           'Qt','GHI','day','SF','DHI'])
        
    return demand, forecast

#create the hourly dataframe
def hour_data(data):
    hours_size = len(data) / 24
    vec = {k : [0] * int(hours_size) for k in range(int(24))}
    
    for t in data.index.tolist():
        #get the hour
        hour_day = t % 24
        #get the day
        day_year = math.floor(t / 24)
        #create data
        vec[hour_day][day_year] = data[t]
        
    return vec


#fix a distribution for each set
def get_best_distribution(vec):
    #get the total hours
    hours = len(vec)
    dist = {}

    #calculate distribution of df
    for i in range(int(hours)):
        dist[i] = best_distribution(vec[i])
    return dist


#get the best distribution
def best_distribution(data):
    #available distributions
    dist_names = [
        "exponweib","norm","weibull_max","weibull_min","pareto", 
        "genextreme","gamma","beta","rayleigh","invgauss",
        "uniform","expon", "lognorm","pearson3","triang"
        ]
    dist_results = []
    params = {}
    #if 0 no distribution, 0 value, example solar generator at night
    if (sum(data) == 0 and np.std(data) == 0):
        best_dist = 'No distribution'
        best_p = None
        params[best_dist] = 0
    else:
        #fit each distribution
        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test and get p-value
            D, p = st.kstest(data, dist_name, args = param)
            dist_results.append((dist_name, p))
    
        # select the best fitted distribution
        best_dist, best_p = (max(dist_results, key = lambda item: item[1]))
        # store the name of the best fit and its p value

    return best_dist, best_p, params[best_dist]


#create stochastic df
def calculate_stochasticity_demand(rand_ob, demand_df, dem_dist):
    for t in demand_df['t']:
        #get the hour for the distribution
        hour = t % 24
        #generate one random number for each hour (demand and forecast)
        obj = generate_random(rand_ob, dem_dist[hour])
        demand_df.loc[t] = [t, obj]
        
    return demand_df
    

def calculate_stochasticity_forecast(rand_ob, forecast_df, wind_dist,
                                     sol_distdni, sol_distdhi, sol_distghi):
    
    for t in forecast_df['t']:
        #get the hour for the distribution
        hour = t % 24
        #generate one random number for each hour (demand and forecast)
        nf_w = generate_random(rand_ob, wind_dist[hour])
        nf_dni = generate_random(rand_ob, sol_distdni[hour])
        nf_dhi = generate_random(rand_ob, sol_distdhi[hour])
        nf_ghi = generate_random(rand_ob, sol_distghi[hour])
        t_ambt = forecast_df['t_ambt'][t]
        Qt = forecast_df['Qt'][t]
        day = forecast_df['day'][t]
        SF = forecast_df['SF'][t]
        forecast_df.loc[t] = [t, nf_dni, t_ambt, nf_w, Qt, nf_ghi, day, SF, nf_dhi]
               
    return forecast_df
    

#generate one random number with distribution
def generate_random(rand_ob, dist):
    if (dist[0] == 'norm'):
        number = rand_ob.dist_norm(dist[2][0], dist[2][1])
    elif (dist[0] == 'uniform'):
        number = rand_ob.dist_uniform(dist[2][0], dist[2][1])
    elif(dist[0] == 'No distribution'):
        number = 0
    elif (dist[0] == 'triang'):
        number = rand_ob.dist_triang(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'exponweib'):
        number = rand_ob.dist_exponweib(dist[2][0], dist[2][1], dist[2][2], dist[2][3])
    elif (dist[0] == 'weibull_max'):
        number = rand_ob.dist_weibull_max(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'weibull_min'):
        number = rand_ob.dist_weibull_min(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'pareto'):
        number = rand_ob.dist_pareto(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'genextreme'):
        number = rand_ob.dist_genextreme(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'gamma'):
        number = rand_ob.dist_gamma(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'beta'):
        number = rand_ob.dist_beta(dist[2][0], dist[2][1], dist[2][2], dist[2][3])
    elif (dist[0] == 'rayleigh'):
        number = rand_ob.dist_rayleigh(dist[2][0], dist[2][1])
    elif (dist[0] == 'invgauss'):
        number = rand_ob.dist_invgauss(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'expon'):
        number = rand_ob.dist_expon(dist[2][0], dist[2][1])
    elif (dist[0] == 'lognorm'):
        number = rand_ob.dist_lognorm(dist[2][0], dist[2][1], dist[2][2])
    elif (dist[0] == 'pearson3'):
        number = rand_ob.dist_pearson3(dist[2][0], dist[2][1], dist[2][2])        
    else:
        number = 0
    return(number)


#generate triangular distribution for parameters like fuel cost
def generate_number_distribution(rand_ob, param, limit):
    #simetric triangular
    a = param * (1 - limit)
    b = param
    c = param * (1 + limit)
    number = rand_ob.dist_triangular(a,b,c)
    return number
