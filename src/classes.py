# -*- coding: utf-8 -*-
import random as random
import numpy as np

class Generator(): #Superclass generators

    def __init__(self, id_gen, tec, br,  area, cost_up, cost_r, cost_s, cost_fopm):
        self.id_gen = id_gen #id of the generator
        self.tec = tec #technology associated to the generator
        self.br = br #brand of the generator
        self.area = area #Generator area
        self.cost_up = cost_up #Investment cost 
        self.cost_r = cost_r #Replacement cost
        self.cost_s = cost_s # Salvament cost 
        self.cost_fopm = cost_fopm #Fixed Operation & Maintenance cost 

class Solar(Generator):
    def __init__(self, id_gen, tec, br, area, cost_up, cost_r, cost_s, cost_fopm, cost_vopm, n, T_noct, G_noct, Ppv_stc, fpv, kt): 
        self.cost_vopm = cost_vopm #Variable Operation & Maintenance cost 
        self.n = n #Number of panels
        self.T_noct = T_noct #Nominal Operating cell Tmperature
        self.G_noct = G_noct #Irradiance operating Normal Condition
        self.Ppv_stc = Ppv_stc #Rated power
        self.fpv = fpv #derating factor
        self.kt = kt #Temperature coefficient
        self.gen_rule = {}
        self.cost_rule = 0
        self.INOCT = 0
        super(Solar, self).__init__(id_gen, tec, br,area, cost_up,  cost_r, cost_s, cost_fopm)

    def Solargeneration(self, t_amb, gt,G_stc): 
    #def Solargeneration(self, t_amb, gt,G_stc,deg): 
            #G_stc: Standar solar radiation
            #Calculate generation over the time
            for t in list(gt.index.values):
                Irad_panel = gt['gt'][t] #irradiance in module W/m2
                if Irad_panel<=0:
                   self.gen_rule[t] = 0
                else:
                    TM = t_amb[t] + (self.INOCT - 20)*(Irad_panel/self.G_noct)
                    self.gen_rule[t] = self.Ppv_stc*(Irad_panel/G_stc)*(1 + self.kt*(TM-25))*self.fpv
                    #year = math.floor(gt['t'][t]/8760) 
                    #self.gen_rule[t] = self.gen_rule[t] * (1- deg)**(year)
            return self.gen_rule
    #calculate operative cost
    def Solarcost(self):
        aux = self.cost_vopm * sum(self.gen_rule.values())
        self.cost_rule = aux
        return self.cost_rule
                               
    # Temperature model
    def Get_INOCT(self, caso = 1 , w = 1):
        """caso 1: direct mount
            caso 2: stand-off
            caso 3: rack mount
            w: distans to mount
            w=1 in > x = 11
            w=3 in > x = 3
            w=6 in > x = -1
        """
        if caso == 1:
            inoct = self.T_noct + 18.0
    
        if caso == 2:
            if w <=1:
                inoct = self.T_noct + 11.0
            elif w <=3:
                inoct = self.T_noct + 3.0
            else:
                inoct = self.T_noct - 1.0
    
        if caso == 3:
            inoct = self.T_noct - 3.0
        
        self.INOCT = inoct
        
        return self.INOCT
        
  

class Eolic(Generator):
    def __init__(self, id_gen, tec, br, area, cost_up, cost_r, cost_s, cost_fopm, cost_vopm, s_in, s_rate, s_out, P_y, n_eq, h):
        self.cost_vopm = cost_vopm #Variable Operation & Maintenance cost 
        self.s_in = s_in #Turbine Minimum Generating Speed (Input Speed)
        self.s_rate = s_rate #Rated speed of the wind turbine
        self.s_out = s_out # Turbine Maximum Generation Speed (Output Speed)
        self.P_y = P_y #Rated power of the wind turbine
        self.n_eq = n_eq #n to calculate the generation curve, usually 1,2 or 3
        self.h = h #height
        self.gen_rule = {}
        self.cost_rule = 0
        super(Eolic, self).__init__(id_gen, tec, br, area, cost_up, cost_r, cost_s, cost_fopm)
    
    def Windgeneration(self, forecastWt, h2, coef_hel): #Wt = wind speed over the time
    #def Windgeneration(self, forecastWt, h2, coef_hel,deg): #Wt = wind speed over the time
        #Calculate generation over the time
        Hellmann = (self.h/h2)**coef_hel
        for t in list(forecastWt.index.values):
            #Calculate Hellmann coefficient
            i = forecastWt[t] * Hellmann
            if i <= self.s_in:
              self.gen_rule[t] = 0
            elif i < self.s_rate:
              self.gen_rule[t] = self.P_y*((i**self.n_eq-self.s_in**self.n_eq)/(self.s_rate**self.n_eq-self.s_in**self.n_eq))
              #year = math.floor(forecastWt['t'][t]/8760) 
              #self.gen_rule[t] = self.gen_rule[t] * (1- deg)**(year)
            elif i <= self.s_out:                  
              self.gen_rule[t] =  self.P_y
              #year = math.floor(forecastWt['t'][t]/8760) 
              #self.gen_rule[t] = self.gen_rule[t] * (1- deg)**(year)
            else:
              self.gen_rule[t] = 0
        return self.gen_rule
    #calculate operative cost
    def Windcost(self):
        aux = self.cost_vopm * sum(self.gen_rule.values())
        self.cost_rule = aux
        return self.cost_rule
                               
class Diesel(Generator):
    def __init__(self, id_gen, tec, br, area, cost_up, cost_r, cost_s, cost_fopm, DG_min, DG_max, f0, f1):
        self.DG_min = DG_min #Minimun generation to active the Diesel
        self.DG_max = DG_max #Rated power, maximum generation
        self.f0 = f0 #fuel consumption curve coefficient
        self.f1 = f1 #fuel consumption curve coefficient
        super(Diesel, self).__init__(id_gen, tec, br, area, cost_up, cost_r,  cost_s, cost_fopm)



class Battery():
    def __init__(self, id_bat, tec, br, efc, efd, eb_zero, soc_max, dod_max, alpha, area, cost_up, cost_fopm, cost_r, cost_s ):
  
        self.id_bat = id_bat #Battery id
        self.tec = tec #Technology associated to the battery, only allowed: "B"
        self.br = br #Brand battery
        self.efc = efc #Charge efficency
        self.efd = efd #Discharge efficency
        self.eb_zero = eb_zero #Energy that the battery has stored at time 0
        self.soc_max = soc_max #Maximum capacity that the battery can storage
        self.dod_max = dod_max #Maximum depth of discharge
        self.alpha = alpha #Energy dissipation rate
        self.area = area #Area
        self.cost_up = cost_up #Investment cost
        self.cost_fopm = cost_fopm #Operation & Maintenance cost
        self.cost_r = cost_r #Replacement cost
        self.cost_s = cost_s #Salvament cost
        self.soc_min = 0 #Minimum level of energy that must be in the battery
        
    def calculatesoc(self): #Calculate soc_min with soc_max and dod_max
        self.soc_min = self.soc_max * (1-self.dod_max)
        return self.soc_min

#solution class to save a solution for two stage approach
class Solution():
    def __init__(self, generators_dict_sol, 
                 batteries_dict_sol, 
                 technologies_dict_sol, 
                 renewables_dict_sol, 
                 results):
        self.generators_dict_sol = generators_dict_sol
        self.batteries_dict_sol = batteries_dict_sol
        self.technologies_dict_sol = technologies_dict_sol
        self.renewables_dict_sol = renewables_dict_sol
        self.results = results
        self.feasible = False


class Random_create():
    #crrate random numbers with seed
    def __init__(self, seed = None):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def create_rand_number(self): 
        number_rand = random.random() 
        return number_rand
    
    def create_rand_list(self, list_rand):
        selection = 0
        selection = random.choice(list_rand)
        return selection
    def create_randint(self, inf, sup):
        selection = 0
        selection = random.randint(inf,sup)
        return selection
    def create_randomsample(self,list_rand, n):
        selection = 0
        selection = random.sample(list_rand,n)
        return selection
    def create_randomshuffle(self,list_rand):
        random.shuffle(list_rand)
        return list_rand
    def create_randomnpnormal(self, means,desv,size):
        selection = 0
        selection = np.random.normal(loc=means, scale=desv, size=size)
        return selection
    def dist_triangular(self,a,b,c):
        s = np.random.triangular(a, b, c, 1)
        number = max(0,s[0])
        return number
    def dist_uniform(self,a,b):
        s = np.random.uniform(a,b, 1)
        number = max(0,s[0])
        return number
    def dist_normal(self,a,b):
        s = np.random.normal(a,b, 1)
        number = max(0,s[0])
        return number