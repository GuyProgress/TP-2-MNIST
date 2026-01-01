# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 04:13:14 2025

@author: phemb
"""

import numpy as np

class Generateur_Donnees:
    
    v = 20
    g = 0.1
    f_v = 0.001
    
    def __init__(self, vitesse_initiale = v, pesanteur = g, frottement_visqueux = f_v):
        
        self._v_0 = vitesse_initiale
        self._g = pesanteur
        self._f_v = frottement_visqueux
        
    def position_virtuelle(self,t):
        
        return np.where(- 0.5*self._g*t**2 + self._v_0 *t >= 0,- 0.5*self._g*t**2 + self._v_0 *t,0)
    
    def position_reelle(self,t):
        
        A = self._v_0/self._f_v + self._g/(self._f_v**2)
        B = - A
        
        return np.where(A+B*np.exp(-self._f_v*t)-(self._g/self._f_v)*t >= 0,A+B*np.exp(-self._f_v*t)-(self._g/self._f_v)*t,0)

    def set_vitesse_initiale(self,vitesse):
        
        self._v_0 = vitesse
        
    def set_pesanteur(self,coefficient):
        
        self._g = coefficient
    
    def set_frottement_visqueux(self,coefficient):
        
        self._f_v = coefficient