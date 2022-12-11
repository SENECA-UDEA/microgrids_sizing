# -*- coding: utf-8 -*-
import random as random
import numpy as np
import scipy.stats as sc 


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
    def __init__(self, id_gen, tec, br, area, cost_up, cost_r, cost_s, cost_fopm,
                 cost_vopm, n, T_noct, G_noct, Ppv_stc, fpv, kt): 
        
        self.cost_vopm = cost_vopm #Variable Operation & Maintenance cost 
        self.n = n #Number of panels
        self.T_noct = T_noct #Nominal Operating cell Tmperature
        self.G_noct = G_noct #Irradiance operating Normal Condition
        self.Ppv_stc = Ppv_stc #Rated power
        self.fpv = fpv #derating factor
        self.kt = kt #Temperature coefficient
        self.gen_rule = {}
        self.cost_rule = 0
        self.inoct = 0
        super(Solar, self).__init__(id_gen, tec, br,area, cost_up,  
                                    cost_r, cost_s, cost_fopm)

    def solar_generation(self, t_amb, irr, g_stc): 
            '''Calculate solar generation at each period'''
            #G_stc: Standar solar radiation
            #Calculate generation over the time
            #irr = total irradiance -> diffuse + horizontal + direct
            for t in list(irr.index.values):
                irad_panel = irr['irr'][t] #irradiance in module W/m2
                
                if irad_panel <= 0:
                   self.gen_rule[t] = 0
                else:
                    t_ref = t_amb[t] + (self.inoct - 20) * (irad_panel / self.G_noct) #reference temperature
                    self.gen_rule[t] = self.Ppv_stc * (irad_panel / g_stc) * (1 + self.kt * (t_ref - 25)) * self.fpv
            
            return self.gen_rule

    #calculate operative cost
    def solar_cost(self):
        aux = self.cost_vopm * sum(self.gen_rule.values())
        self.cost_rule = aux
        return self.cost_rule
                               
    # Temperature model
    def get_inoct(self, caso = 1, w = 1):
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
            if w <= 1:
                inoct = self.T_noct + 11.0
            elif w <= 3:
                inoct = self.T_noct + 3.0
            else:
                inoct = self.T_noct - 1.0
        if caso == 3:
            inoct = self.T_noct - 3.0
        
        self.inoct = inoct
        return self.inoct
        
  
class Eolic(Generator):
    def __init__(self, id_gen, tec, br, area, cost_up, cost_r, cost_s, cost_fopm,
                 cost_vopm, s_in, s_rate, s_out, P_y, n_eq, h):
        self.cost_vopm = cost_vopm #Variable Operation & Maintenance cost 
        self.s_in = s_in #Turbine Minimum Generating Speed (Input Speed)
        self.s_rate = s_rate #Rated speed of the wind turbine
        self.s_out = s_out # Turbine Maximum Generation Speed (Output Speed)
        self.P_y = P_y #Rated power of the wind turbine
        self.n_eq = n_eq #n to calculate the generation curve, usually 1,2 or 3
        self.h = h #height
        self.gen_rule = {}
        self.cost_rule = 0
        super(Eolic, self).__init__(id_gen, tec, br, area, cost_up,
                                    cost_r, cost_s, cost_fopm)
    
    def eolic_generation(self, forecastWt, h2, coef_hel): 
        #Wt = wind speed over the time
        #Calculate generation over the time
        HELLMANN = (self.h / h2) ** coef_hel
        for t in list(forecastWt.index.values):
            #Calculate Hellmann coefficient
            i = forecastWt[t] * HELLMANN
            
            if i <= self.s_in:
              self.gen_rule[t] = 0
            elif i < self.s_rate:
              self.gen_rule[t] = self.P_y * ((i**self.n_eq - self.s_in**self.n_eq)
                                            / (self.s_rate**self.n_eq - self.s_in**self.n_eq))
            elif i <= self.s_out:                  
              self.gen_rule[t] =  self.P_y
            else:
              self.gen_rule[t] = 0
       
        return self.gen_rule
    
    #calculate operative cost
    def eolic_cost(self):
        aux = self.cost_vopm * sum(self.gen_rule.values())
        self.cost_rule = aux
        return self.cost_rule
    
                           
class Diesel(Generator):
    def __init__(self, id_gen, tec, br, area, cost_up, cost_r, cost_s, 
                 cost_fopm, DG_min, DG_max, f0, f1):
        self.DG_min = DG_min #Minimun generation to active the Diesel
        self.DG_max = DG_max #Rated power, maximum generation
        self.f0 = f0 #fuel consumption curve coefficient
        self.f1 = f1 #fuel consumption curve coefficient
        super(Diesel, self).__init__(id_gen, tec, br, area, cost_up,
                                     cost_r, cost_s, cost_fopm)


class Battery():
    def __init__(self, id_bat, tec, br, efc, efd, eb_zero, soc_max, dod_max, alpha, 
                 area, cost_up, cost_fopm, cost_r, cost_s, cost_vopm):  
        
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
        self.cost_vopm = cost_vopm #variable cost
        
    def calculate_soc(self): #Calculate soc_min with soc_max and dod_max
        self.soc_min = self.soc_max * (1 - self.dod_max)
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


class RandomCreate():
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

    def create_rand_int(self, inf, sup):
        selection = 0
        selection = random.randint(inf, sup)
        return selection

    def create_rand_sample(self, list_rand, n):
        selection = 0
        selection = random.sample(list_rand, n)
        return selection

    def create_rand_shuffle(self, list_rand):
        random.shuffle(list_rand)
        return list_rand

    def create_rand_p_normal(self, means, desv, size):
        selection = 0
        selection = np.random.normal(loc = means, scale = desv, size = size)
        return selection
    
    #distributions for stochasticity
    def dist_triang(self, a, b, c):
        rand_generation = sc.triang.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_uniform(self, a, b):
        rand_generation = sc.uniform.rvs(loc = a, scale = b, size = 1,random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_norm(self, a, b):
        rand_generation = sc.norm.rvs(loc = a, scale = b, size = 1,random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_exponweib(self,a, b, c, d):
        rand_generation = sc.exponweib.rvs(a, b, loc = c, scale = d, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_weibull_max(self, a, b, c):
        rand_generation = sc.weibull_max.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_weibull_min(self, a, b, c):
        rand_generation = sc.weibull_min.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_pareto(self, a, b, c):
        rand_generation = sc.pareto.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_genextreme(self, a, b, c):
        rand_generation = sc.genextreme.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_gamma(self, a, b, c):
        rand_generation = sc.gamma.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_beta(self, a, b, c, d):
        rand_generation = sc.beta.rvs(a, b, loc = c, scale = d, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_rayleigh(self, a, b):
        rand_generation = sc.rayleigh.rvs(loc = a, scale = b, size = 1,random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_invgauss(self, a, b, c):
        rand_generation = sc.triang.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_expon(self, a, b):
        rand_generation = sc.expon.rvs(loc = a, scale = b, size = 1,random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_lognorm(self, a, b, c):
        rand_generation = sc.lognorm.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    def dist_pearson3(self, a, b, c):
        rand_generation = sc.pearson3.rvs(a, loc = b, scale = c, size = 1, random_state = self.seed)
        number = max(0, rand_generation[0])
        return number

    #triangular distribution for parameters
    def dist_triangular(self, a, b, c):
        rand_generation = np.random.triangular(a, b, c, 1)
        number = max(0, rand_generation[0])
        return number