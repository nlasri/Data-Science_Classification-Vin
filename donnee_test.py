# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:47:24 2020

@author: Andréa WAUTERS et Narjisse LASRI
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import perceptron as pt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from time import perf_counter
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import pandas as pd

#n_samples : le nombre d'échantillons qui composeront notre couple X,y = 1500
#n_samples : le nombre d'échantillons qui composeront notre couple X,y = 1500
#n_features : Le nombre total de fonctionnalités ici deux : n_redundant et n_informative voir après
#n_informative : Le nombre d'éléments informatifs, on en souhaite qu'une pour pouvoir représenter les données facilement donc 1
#n_redundant : nombre de carractéristiques redondantes doit être = 0 on veut qu'elles soient toutes différentes sinon on aurait simplement une droite
#class_sep : Le facteur qui multiplie la taille de l'hypercube : permet que les données des deux classes soient plus ou moins éloignées
#n_cluster_per_class : le nombre de cluster par classe doit être = 1 pour suivre notre modèle (surtout pour y)

X, y = make_classification(n_samples=1500, n_features=2, n_informative=1, n_redundant=0, class_sep=1, n_clusters_per_class=1)


#Y est crée avec les valeurs 0 ou 1, pour suivre notre modèle on remplace les 0 par des -1
y = np.where(y==0, -1, y)


#On créé l'affichage en choissisant les deux colones de X en abcisse et ordonnée, c=y nous permet de donner deux couleurs différentes aux deux classes 
plt.scatter(X[:, 0], X[:, 1], marker='p', c=y, s=20, edgecolor='g')
plt.show()



def test_perceptron():
    #On choisit n=0.1, e=0.1 et max_iteration=100
    list_classifieur = [pt.Perceptron(0.1, 0.1, 100), Perceptron(tol=0.1, random_state=0, max_iter=400, alpha=0.1)]
    
    temp = np.empty([2, 3]) #Tableau récapitulatif des scores de chaque classifieur
    i=0 
    for classifieur in list_classifieur: #On parcourt les différents classifieurs
        model = list_classifieur[i] 
        
        #On teste les classifieurs : recall, score f1, accuracy
        score = cross_validate(model, X, y, cv=StratifiedKFold(n_splits=10, random_state=int(perf_counter ()*100), shuffle=True), scoring=dict(recal=make_scorer(recall_score), f1 =make_scorer(f1_score), accur = make_scorer(accuracy_score)), return_train_score=False)
        
        recall = score['test_recal'].mean() #Score moyen de recall du classifieur
        temp[i][0] = recall 
        
        f1 = score['test_f1'].mean() #Score moyen de score f1 du classifieur
        temp[i][1] = f1
    
        accuracy = score['test_accur'].mean() #Score moyen de accuracy du classifieur
        temp[i][2] = accuracy
        i=i+1
       
    #On crée un tableau avec des noms de colonnes et de lignes pour temp pour permettre une meilleure lecture
    total = pd.DataFrame(temp, index = ['Notre perceptron', 'SKlearn'], columns = ['recall','f1','accuracy'])
    print("\nscore total avec validation croisee")
    print(total)
        

test_perceptron()

 