# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:04:49 2022

@author: pmayaduque
"""

class Generator():

    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max):
        self.id_gen = id_gen
        self.tec = tec
        self.br = br
        self.va_op = va_op
        self.area = area
        self.cost_up = cost_up
        self.cost_r = cost_r
        self.cost_om = cost_om
        self.cost_s = cost_s
        self.c_min = c_min
        self.c_max = c_max

class Solar(Generator):

    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s,  c_min, c_max, ef, G_test, R_test):
        self.ef = ef
        self.G_test = G_test
        self.R_test = R_test
        self.gen_rule = {}
        super(Solar, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max)
    
    def Solargeneration(self, forecastGt):
            t = 0
            for i in forecastGt:
                self.gen_rule[t] = self.ef * self.G_test * (forecastGt[t]/self.R_test) 
                t += 1
            return self.gen_rule


                                  
class Eolic(Generator):
    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max, ef, n, s, p, w_in, w_rate, w_out, n_eq):
        self.ef = ef
        self.n = n
        self.s = s
        self.p = p
        self.w_in = w_in
        self.w_rate = w_rate
        self.w_out = w_out
        self.n_eq = n_eq
        self.gen_rule = {}
        super(Eolic, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max)
    
    def Windgeneration(self, forecastWt):
            t = 0
            for i in forecastWt:
                if i <= self.w_in:
                  self.gen_rule[t] = 0
                elif i < self.w_rate:
                  self.gen_rule[t] =  self.c_max*((i**self.n_eq-self.w_in**self.n_eq)/(self.w_rate**self.n_eq-self.w_in**self.n_eq))
                elif i <= self.w_out:                  
                  self.gen_rule[t] = self.c_max
                else:
                  self.gen_rule[t] = 0
                t += 1
            return self.gen_rule


                               
class Diesel(Generator):
    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s,  c_min, c_max, ef, G_min, G_max):
        self.ef = ef
        self.G_min = G_min
        self.G_max = G_max
        self.gen_rule = {}
        super(Diesel, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max)
    
    def Dieselgeneration(self, forecast):
            t = 0
            for i in forecast:
                self.gen_rule[t] = self.c_max
                t += 1
            return self.gen_rule


class Battery():
    def __init__(self, id_bat, tec, br, efc, efd, eb_zero, soc_max, dod_max, alpha, va_op, area, cost_up, cost_r, cost_om, cost_s ):
  
        self.id_bat = id_bat
        self.tec = tec
        self.br = br
        self.efc = efc
        self.efd = efd
        self.eb_zero = eb_zero
        self.soc_max = soc_max
        self.dod_max = dod_max
        self.alpha = alpha
        self.va_op = va_op
        self.area = area
        self.cost_up = cost_up
        self.cost_r = cost_r
        self.cost_om = cost_om
        self.cost_s = cost_s
        self.soc_min = 0

    def calculatesoc(self):
        self.soc_min = self.soc_max * (1-self.dod_max)
        return self.soc_min

