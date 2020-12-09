# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:47:24 2020

@author: Andréa WAUTERS et Narjisse LASRI
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Perceptron(BaseEstimator, ClassifierMixin):

    #Init : Initalisation du perceptron 
    #n : le pas d'ajustement de la courbe de séparation
    #e : l'erreur : distance cumulée du point par rapport à la courbe max acceptable pour arrêter l'ajustement de la courbe de séparation
    #max_iteration : le nombre d'itérations maximales accaptées pour établir la courbe de séparation
    #--> courbe de séparation des données    
    def __init__(self, n , e, max_iteration): 
        self.n=n
        self.e=e
        self.max_iteration=max_iteration
         
    # fonction qui retourne -1 si le point est en dessous de la courbe et 1 s'il est au dessus de la courbe
    def signe (self,x):
        signe = np.dot(x,self.coef_dir[1:])+self.coef_dir[0]
        if signe > 0:   #au dessus de la courbe donc le type est de 1
            return 1
        else:
            return -1   #en dessous de la courbe donc le type est de -1


    #La fonction fit permet d'établir la courbe de séparation des données par une analyse de données
    def fit(self, X, y):
                 
        #on s'assure que les données soient bien des tableaux numpy : 
        X=np.array(X)    
        y=np.array(y)

        #on récupère le nombre de colones qui définissent une ligne
        taille=X.shape[1]

        #On peut ainsi initialiser les coef dir de notre courbe de séparation 
        self.coef_dir=np.ones(taille+1) # +1 pour le coeffficient O0
  
        #On passe au mélange des données :        
        X=np.c_[X,y]             #La fct c_ permet de concatener X et y (en terme de colone)
        np.random.shuffle(X)     #Shuffle mélange les lignes
        y=X[:,taille]            #On récupère les nouvelles valeurs de y
        X=np.delete(X,taille,1)  #On récupère les nouvelles valeurs de x (en supprimant la colone y)

        #Première boucle qui permet d'arrêter la fct si on dépasse le maximum d'itérations
        for i in range (0,self.max_iteration):  #maximum d'itérations            
            distance = 0                        #Distance (cumulée d'erreur) remise à 0 pour chaque nouveau parcours des données
        
            #Deuxième boucle qui permet de parcourir les données et leur type
            for donnee,types in zip(X,y): 
                
                signe = self.signe(donnee)   #On récupère le type prédit pour la donnée traitée par rapport à la courbe de séparation
                if signe != types:           #On vérifie que ce type correspond au type réel 
                    distance += types*(np.dot(donnee,self.coef_dir[1:])+self.coef_dir[0])    #On incrémante l'erreur avec la distance à le courbe de la donnée traitée
                    if(np.abs(distance)<self.e):  #Si celle-ci est inférieure à celle prévue nous avons fini notre ajustement on peut arrêter le Fit
                        return
                    self.coef_dir[1:]+= self.n*donnee*types #On incrémante alors nos coeff dir pour ajuster la courbe de prédiction
                    self.coef_dir[0]+= self.n*types         #Sans oublier d'incrémenter O0
       
        return self


    #Fonction de prediction du type d'un ou plusieurs points X        
    def predict(self, X):            
        X=np.array(X)  #on s'assure que X soit un tableau numpy
        y=[]           #on initialise y qui contiendra la ou les prédictions 

        #Pour chaque valeur de X on ajoute la prédiction dans Y
        for valeur in X:            
            y.append(self.signe(valeur))

        #On retourne Y             
        return y