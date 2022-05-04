# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:04:49 2022

@author: pmayaduque
"""

class Generator(): #Superclass generators

    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s):
        self.id_gen = id_gen #id of the generator
        self.tec = tec #technology associated to the generator
        self.br = br #brand of the generator
        self.va_op = va_op #Operative cost 
        self.area = area #Generator area
        self.cost_up = cost_up #Investment cost
        self.cost_r = cost_r #Replacement cost
        self.cost_om = cost_om #Operation & Maintenance cost
        self.cost_s = cost_s # Salvament cost 


class Solar(Generator):

    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, ef, G_test, R_test, n):
        self.ef = ef #Efficiency 
        self.G_test = G_test #Rated power of the solar panel in standard test condition
        self.R_test = R_test #Global irradiance in standard test condition of the PV cell 
        self.n = n #Number of panels
        self.gen_rule = {}
        super(Solar, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s)
    
    def Solargeneration(self, forecastGt): #Gt = irradiance data over the time   
            #Calculate generation over the time
            for t in list(forecastGt.index.values):
                self.gen_rule[t] = self.n * self.ef * self.G_test * (forecastGt[t]/self.R_test) 
            return self.gen_rule


class Eolic(Generator):
    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, ef, s_in, s_rate, s_out, rp, n, n_eq):
        self.ef = ef #Efficiency
        self.s_in = s_in #Turbine Minimum Generating Speed (Input Speed)
        self.s_rate = s_rate #Rated speed of the wind turbine
        self.s_out = s_out # Turbine Maximum Generation Speed (Output Speed)
        self.rp = rp #Rated power of the wind turbine
        self.n = n #number of wind turbines
        self.n_eq = n_eq #n to calculate the generation curve, usually 1,2 or 3
        self.gen_rule = {}
        super(Eolic, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s)
    
    def Windgeneration(self, forecastWt): #Wt = wind speed over the time
        #Calculate generation over the time
            for t in list(forecastWt.index.values):
                i = forecastWt[t]
                if i <= self.s_in:
                  self.gen_rule[t] = 0
                elif i < self.s_rate:
                  self.gen_rule[t] =  self.n * self.rp*((i**self.n_eq-self.s_in**self.n_eq)/(self.s_rate**self.n_eq-self.s_in**self.n_eq))
                elif i <= self.s_out:                  
                  self.gen_rule[t] = self.n * self.rp
                else:
                  self.gen_rule[t] = 0
            return self.gen_rule


                               
class Diesel(Generator):
    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, ef, G_min, G_max, n):
        self.ef = ef #efficiency
        self.G_min = G_min #Minimun generation to active the Diesel
        self.G_max = G_max #Rated power, maximum generation
        self.n = n #Number of diesel generators
        self.gen_rule = {}
        super(Diesel, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s)
    
    def Dieselgeneration(self, demand): #Calculate maximum generation capacity
            for t in list(demand.index.values):
                self.gen_rule[t] = self.n * self.G_max
            return self.gen_rule


class Battery():
    def __init__(self, id_bat, tec, br, efc, efd, eb_zero, soc_max, dod_max, alpha, va_op, area, cost_up, cost_r, cost_om, cost_s ):
  
        self.id_bat = id_bat #Battery id
        self.tec = tec #Technology associated to the battery, only allowed: "B"
        self.br = br #Brand battery
        self.efc = efc #Charge efficency
        self.efd = efd #Discharge efficency
        self.eb_zero = eb_zero #Energy that the battery has stored at time 0
        self.soc_max = soc_max #Maximum capacity that the battery can storage
        self.dod_max = dod_max #Maximum depth of discharge
        self.alpha = alpha #Energy dissipation rate
        self.va_op = va_op #Operational cost
        self.area = area #Area
        self.cost_up = cost_up #Investment cost
        self.cost_r = cost_r #Replacement cost
        self.cost_om = cost_om #Operation & Maintenance cost
        self.cost_s = cost_s #Salvament cost
        self.soc_min = 0 #Minimum level of energy that must be in the battery
        #TODO: check if the battery need n
        
    def calculatesoc(self): #Calculate soc_min with soc_max and dod_max
        self.soc_min = self.soc_max * (1-self.dod_max)
        return self.soc_min

