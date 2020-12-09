# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:47:24 2020

@author: Andréa WAUTERS et Narjisse LASRI
"""

import pandas as pd
import vin as pj
import perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
import numpy as np
import perceptron as pt
from sklearn.linear_model import Perceptron



red_wines = pd.read_csv('red_wines.csv', sep = ',', engine = 'python')

#-------------------------PARTIE 2 --------------------------------------------

#-------------------------Classes ---------------------------------------------

def nom_classes():
    print("Les classes sont :")
    pj.classe(red_wines)

#nom_classes()

print(red_wines)

#-------------------------Valeurs aberrantes ----------------------------------

#analyse des valeurs >14 pour le PH
red_wines.loc[red_wines['pH'] > 14,'pH'] = np.nan #les valeurs >14 deviennent nulles

#print(red_wines['pH'].isna().sum()) #on se rend compte que 18 valeurs ont été supprimées

#red_wines.hist() # permet de se rendre compte des variables qui ont une allure gaussienne en affichant leur graph

#remplacement des valeurs, des coubes gaussiennes, aberrantes par null
red_wines=pj.nettoyage_ecart_type(3,'pH',red_wines)
red_wines=pj.nettoyage_ecart_type(3,'fixed acidity',red_wines)
red_wines=pj.nettoyage_ecart_type(3,'density',red_wines)
red_wines=pj.nettoyage_ecart_type(3,'sulphates',red_wines)
red_wines=pj.nettoyage_ecart_type(3,'volatile acidity',red_wines)

#Proportion des lignes nulles
print("Propotion de lignes à supprimer ou remplacer par la moyenne : " + str (pj.proportion_val_null(red_wines))) 

#on peut donc decider de supprimer ces valeurs nulles (moins de precision) ou remplacer par la moyenne (+ veracité)
red_wines = red_wines.dropna() #supprimer le lignes avec des valeurs non numériques

"""#ou deuxième choix remplacer par la moyenne de chaque colone
red_wines = remplacer_val_moy(red_wines,'alcohol')
red_wines = remplacer_val_moy(red_wines,'chlorides')
red_wines = remplacer_val_moy(red_wines,'citric acid')
red_wines = remplacer_val_moy(red_wines,'pH')
red_wines = remplacer_val_moy(red_wines,'free sulfur dioxide')
red_wines = remplacer_val_moy(red_wines,'fixed acidity')
red_wines = remplacer_val_moy(red_wines,'residual sugar')
red_wines = remplacer_val_moy(red_wines,'density')
red_wines = remplacer_val_moy(red_wines,'sulphates')
red_wines = remplacer_val_moy(red_wines,'total sulfur dioxide')
red_wines = remplacer_val_moy(red_wines,'volatile acidity')
red_wines = red_wines.dropna() #permet de supprimer les lignes n'ayant pas d'indice de qualité """


print("\n")


#-------------------------Correlation ----------------------------------------- 

#Un graphique de nuage de points met en valeur la corrélation
#La fonction graph_corr permet d'afficher des graphiques de nuages de points entre les attributs
#pj.graph_corr(red_wines)


#Le choix 1 permet de voir les attributs corrélés entre eux
#Le choix 2 permet de supprimer les attributs corrélés entre eux
def test_correlation(choix):
    if choix == 1:
        pj.liste_corr(red_wines, 0.5)
    elif choix == 2:
        pj.correlation(red_wines, 0.5, 'pearson')
        pj.correlation(red_wines, 0.5, 'spearman')
       
test_correlation(2)

#-------------------------Proportion des qualités------------------------------
quality=red_wines['quality'].value_counts()
quality_total = red_wines['quality'].shape[0]
print("Proportion de vin de bonne qualité : " +str(float(quality[1])*100/quality_total))
print("Proportion de vin de mauvaise qualité : " + str (float(quality[-1])*100/quality_total))
    
 
#-------------------------Centrer et reduire----------------------------------- 

#Les données vont être centrées et réduites (sauf la colonne qualité). La moyenne est d'environ 0 après cette étape
red_wines = pj.centrer_reduire(red_wines)
print("Moyenne de la fonction :")
print(red_wines.mean(axis=0))

#-------------------------Test des classifieurs--------------------------------

#Le test du classifieur se fait avec régression logistique, machines à vecteurs supports, analyse discriminante linéaire, analyse discriminante quadratique, k-plus proches voisins, arbres de d´ecision
#Les 2 derniers classifieurs testés sont le perceptron créé par notre binôme et le perceptron de sklearn, ce qui permet de les comparer
#Le choix 1 permet de tester les classifieurs avec la validation croisée
#Le choix 2 permet de tester les classifieurs avec une simple division des données entre test (1/3) et apprentissage (2/3)
def test_classifieur(choix):
    list_classifieur = [LogisticRegression(), svm.SVC(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), KNeighborsClassifier(n_neighbors=3), tree.DecisionTreeClassifier(random_state=0), pt.Perceptron(0.1, 0.1, 400), Perceptron(tol=1e-3, random_state=0)]
    if choix == 1:
        pj.test_classifieurs(list_classifieur, red_wines)
        print("\n")
    elif choix == 2:
        pj.test_classifieurs2(list_classifieur, red_wines)
        print("\n")

test_classifieur(1)

#-------------------------Test du Perceptron créé------------------------------

print("Nous cherchons le meilleur max d'itération. Nous fixons l'erreur à 0.1 et le pas à 0.15 ")
#pj.test_perceptron_iteration(0.15, 0.1)

print("Nous cherchons le meilleur taux d'apprentissage. Nous fixons l'erreur à 0.1 et le nombre d'iteration max à 400 ")
#pj.test_perceptron_taux_apprentissage(0.1, 400)





