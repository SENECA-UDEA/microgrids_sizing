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
        super(Solar, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max)
                                          
class Eolic(Generator):
    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max, ef, n, s, p, w_min, w_a, w_max):
        self.ef = ef
        self.n = n
        self.s = s
        self.p = p
        self.w_min = w_min
        self.w_a = w_a
        self.w_max = w_max
        super(Eolic, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max)
                                        
class Diesel(Generator):
    def __init__(self, id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s,  c_min, c_max, ef, G_min, G_max):
        self.ef = ef
        self.G_min = G_min
        self.G_max = G_max
        super(Diesel, self).__init__(id_gen, tec, br, va_op, area, cost_up, cost_r, cost_om, cost_s, c_min, c_max)

class Battery():
    def __init__(self, id_bat, tec, br, efc, efd, eb_zero, soc_max, soc_min, alpha, va_op, area, cost_up, cost_r, cost_om, cost_s ):
  
        self.id_bat = id_bat
        self.tec = tec
        self.br = br
        self.efc = efc
        self.efd = efd
        self.eb_zero = eb_zero
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.alpha = alpha
        self.va_op = va_op
        self.area = area
        self.cost_up = cost_up
        self.cost_r = cost_r
        self.cost_om = cost_om
        self.cost_s = cost_s
