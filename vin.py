# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:47:24 2020

@author: Andréa WAUTERS et Narjisse LASRI
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from time import perf_counter
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import perceptron as pt



red_wines = pd.read_csv('red_wines.csv', sep = ',', engine = 'python')


#-------------------Classes -------------------------

#Cette fonction permet de trouver les classes de la base de données du projet
def classe(dataset):
    temp = red_wines.groupby('quality').nunique()
    print(temp)
    

#-------------------Valeurs aberrantes -------------------------

#Fontion de remplacement des valeurs supérieures et inférieures à + ou - nb_ecart type * moy par null
def nettoyage_ecart_type (nb_ecart_type, column, matrice):
    ecart_type = matrice[column].std()  #calcul écart_type
    moyenne = matrice[column].mean()    #calcul moyenne
    matrice.loc[matrice[column] > moyenne + nb_ecart_type*ecart_type,column] = np.nan  #remplacement des valeurs au dessus de la borne supérieure
    matrice.loc[matrice[column] < moyenne - nb_ecart_type*ecart_type,column] = np.nan  #remplacement des valeurs en dessous de la borne inférieure
    return matrice

#Fontion de calcul de la proportion de valeurs nulles dans une matrice
def proportion_val_null(matrice):
    return 100 - matrice.dropna().shape[0] *100 / matrice.shape[0]

#Fontion qui remplace les valeurs nulles d'une colone par la moyenne de celle-ci
def remplacer_val_moy(matrice,column):
    moyenne = matrice[column].mean()  #calculer moyenne
    matrice = matrice.fillna(moyenne) #remplacer les valeurs nulles par la moyenne des valeurs
    return matrice


#-------------------Correlation -----------------------------------------------

#Cette fonction affiche un graphique de nuage de points entre les attributs 2 à 2
def graph_corr(dataset):
    sns.pairplot(dataset, height=2);
    plt.show()


#Cette fonction permet de lister tous les attributs corrélés
def liste_corr(dataset, threshold):
    list_attributes_corr = list() #Liste des couples d'attributs fortement corrélés
    corr_matrix = dataset.corr(method='spearman') #matrice de corrélation avec la méthode Spearman
    for i in range(len(corr_matrix.columns)):
        for j in range(i): # on parcourt le triangle supérieur de la matrice
            if (corr_matrix.iloc[i, j] >= threshold):
                attributes_corr = corr_matrix.columns[i] +" et "+corr_matrix.columns[j] # On prend les attributs corrélés
                list_attributes_corr.append(attributes_corr)
    print("attributs correles : ")
    print(list_attributes_corr) 
    print("\n")
    

#Cette fonction permet de trouver les attributs corrélés de façon linéaire ou non puis de les supprimer
def correlation(dataset, threshold, methode):
    col_supp = list() # Liste des colonnes à supprimer
    list_attributes_corr = list() #Liste des couples d'attributs fortement corrélés
    corr_matrix = dataset.corr(method = methode) #matrice de corrélation avec la méthode Pearson ou Spearman
    for i in range(len(corr_matrix.columns)):
        for j in range(i): # On parcourt le triangle supérieur de la matrice
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_supp):
                col_name = corr_matrix.columns[i] # On prend le nom de la colonne
                attributes_corr = corr_matrix.columns[i] +" et "+corr_matrix.columns[j] # On prend les attributs corrélés
                list_attributes_corr.append(attributes_corr)
                col_supp.append(col_name)
                if col_name in dataset.columns: 
                    del dataset[col_name] # On supprime la colonne
    
    if methode == 'pearson':
        print("attributs correles lineairement : ")
    else:
        print("attributs correles non lineairement : ")
    print(list_attributes_corr)
    print("Suppression des attributs : ")
    print(col_supp)
    print("\n")


#Cette fonction permet d'évaluer grâce à un graphe l'ordre de la relation entre 2 attributs en modifiant la valeur de l'ordre et en cherchant visuellement quelle courbe s'ajuste le mieux aux données   
def relation_attributs(dataset, column1, column2, valeur_ordre):
    sns.regplot(x=dataset[column1], y=dataset[column2], fit_reg=True, order=valeur_ordre)
    plt.show()


#-------------------------Proportion des qualités------------------------------

quality=red_wines['quality'].value_counts()
quality_total = red_wines['quality'].shape[0]
print("Proportion de vin de bonne qualité : " +str(float(quality[1])*100/quality_total))
print("Proportion de vin de mauvaise qualité : " + str (float(quality[-1])*100/quality_total))
    
 
#------------------------Centrer et reduire-----------------------------------

#Cette fonction permet de centrer et réduire les données
def centrer_reduire(dataset):
    dataset2 = dataset.drop(columns=['quality']) #On isole les données à centrer/réduire en enlevant la colonne qualité
    dataset2 = (dataset2 - dataset2.mean()) / (dataset2.std()) #On enlève la moyenne pour centrer et on divise par l'écart type pour réduire
    dataset2['quality'] = dataset['quality'] # On remet la colonne qualité aux données
    return dataset2


#-------------------------Test des classifieurs-------------------------------
  
#Division par validation croisée
def test_classifieurs(list_classifieur, dataset):
    df = dataset
    connu = df.drop(['quality'], axis=1) #Tableau qui contient les données connues
    prediction = df['quality'] #Tableau qui contient les données à prédire (les classes)
    temp = np.empty([8, 3]) #Tableau récapitulatif des scores de chaque classifieur
    i=0
    for classifieur in list_classifieur: #On parcourt les différents classifieurs
        model = list_classifieur[i] 
        #On teste les classifieurs avec recall, score f1, accuracy
        score = cross_validate(model, connu, prediction, cv=StratifiedKFold(n_splits=10, random_state=int(perf_counter ()*100), shuffle=True), scoring=dict(recal=make_scorer(recall_score), f1 =make_scorer(f1_score), accur = make_scorer(accuracy_score)), return_train_score=False)
        
        recall = score['test_recal'].mean() #Score moyen de recall du classifieur
        temp[i][0] = recall 
        
        f1 = score['test_f1'].mean() #Score moyen de score f1 du classifieur
        temp[i][1] = f1
    
        accuracy = score['test_accur'].mean() #Score moyen de accuracy du classifieur
        temp[i][2] = accuracy
        i=i+1
       
    #On crée un tableau avec des noms de colonnes et de lignes pour temp pour permettre une meilleure lecture
    total = pd.DataFrame(temp, index = ['LogisticRegression', 'svm.SVC', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'KNeighborsClassifier', 'tree.DecisionTreeClassifier', 'Notre perceptron', 'SKlearn'], columns = ['recall','f1','accuracy'])
    print("\nscore total avec validation croisée")
    print(total)
    

#Division des données 2/3 et 1/3
def test_classifieurs2(list_classifieur, dataset):
    df = dataset
    X = df.drop(['quality'], axis=1) #Tableau qui contient les données connues
    y = df['quality']  #Tableau qui contient les données à prédire (les classes)
    temp = np.empty([8, 3]) #Tableau récapitulatif des scores de chaque classifieur
    i=0
    for classifieur in list_classifieur: #On parcourt les différents classifieurs
        
        #On divise les données en données d'apprentissage (2/3) et données de test (1/3 = 0.33)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        #On entraine le classifieur puis on le teste
        model = list_classifieur[i] 
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        
        recall = recall_score(y_test, prediction) #Score recall du classifieur
        temp[i][0] = recall
        
        
        f1 = f1_score(y_test, prediction) #Score f1 du classifieur
        temp[i][1] = f1
        
    
        accuracy = accuracy_score(y_test, prediction) #Score accuracy du classsifieur
        temp[i][2] = accuracy
        i=i+1
        
    #On crée un tableau avec des noms de colonnes et de lignes pour temp pour permettre une meilleure lecture    
    total = pd.DataFrame(temp, index = ['LogisticRegression', 'svm.SVC', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'KNeighborsClassifier', 'tree.DecisionTreeClassifier', 'Notre perceptron', 'SKlearn'], columns = ['recall','f1','accuracy'])
    print("\nscore total avec division des données")
    print(total)


#-------------------------Test du Perceptron créé------------------------------
    
#Fonction pour définir le taux d'apprentissage
def test_perceptron_taux_apprentissage(e,max_iteration):
    df = red_wines
    connu = df.drop(['quality'], axis=1) #Tableau qui contient les données connues
    prediction = df['quality'] #Tableau qui contient les données à prédire (les classes)
    temp = np.empty([20, 4]) #Tableau récapitulatif des scores de chaque taux d'apprentissage
    i=0 #variable pour parcourir temp
    for j in np.arange(0, 1, 0.05): #Le taux d'apprentissage varie de 0 à 1
        temp[i][0]=j #On remplie la colonne contenant le taux d'aprentissage
        model = pt.Perceptron(j, e, max_iteration) #Le classifieur utilisé est le perceptron créé
        
        #On teste le classifieur avec recall, score f1, accuracy
        score = cross_validate(model, connu, prediction, cv=StratifiedKFold(n_splits=10, random_state=int(perf_counter ()*100), shuffle=True), scoring=dict(recal=make_scorer(recall_score), f1 =make_scorer(f1_score), accur = make_scorer(accuracy_score)), return_train_score=False)
        
        recall = score['test_recal'].mean()
        temp[i][1] = recall
        
        f1 = score['test_f1'].mean()
        temp[i][2] = f1
    
        accuracy = score['test_accur'].mean()
        temp[i][3] = accuracy
        i=i+1
        
    #On crée un tableau avec des noms de colonnes et de lignes pour temp pour permettre une meilleure lecture  
    total = pd.DataFrame(temp, columns = ['taux d apprentissage','recall','f1','accuracy'])
    print("\nScore total pour les taux d'apprentissage")
    print(total)
    
    print("Le meilleur taux d'apprentissage est : ")
    print(total.loc[total['accuracy'].idxmax(), 'taux d apprentissage']) #On prend le taux d'apprentissage avec le meilleur score accuracy
    
    #On crée un graphique pour représenter l'évolution du score accuracy en fonction du taux d'apprentissage
    X = total.loc[:, ['taux d apprentissage']]
    y= total.loc[:, ['accuracy']]
    plt.title("Evolution du score accuracy en fonction du taux d'apprentissage")
    plt.plot(X,y)  # on crée la courbe
    plt.ylabel('score accuracy')
    plt.xlabel("taux d'apprentissage")
    
#Fonction pour définir le max d'itérations
def test_perceptron_iteration(n, e):
    df = red_wines
    connu = df.drop(['quality'], axis=1) #Tableau qui contient les données connues
    prediction = df['quality'] #Tableau qui contient les données à prédire (les classes)
    temp = np.empty([20, 4]) #Tableau récapitulatif des scores de chaque max d'itérations
    i=0 #variable pour parcourir temp
    for j in np.arange(0, 2000, 100): #Le max d'itérations varie de 0 à 2000
        temp[i][0]=j #On remplie la colonne contenant la valeur du max d'itération
        model = pt.Perceptron(n, e, j) #Le classifieur utilisé est le perceptron créé
        
        #On teste le classifieur avec recall, score f1, accuracy
        score = cross_validate(model, connu, prediction, cv=StratifiedKFold(n_splits=10, random_state=int(perf_counter ()*100), shuffle=True), scoring=dict(recal=make_scorer(recall_score), f1 =make_scorer(f1_score), accur = make_scorer(accuracy_score)), return_train_score=False)
        
        recall = score['test_recal'].mean()
        temp[i][1] = recall
        
        f1 = score['test_f1'].mean()
        temp[i][2] = f1
    
        accuracy = score['test_accur'].mean()
        temp[i][3] = accuracy
        i=i+1
    
    #On crée un tableau avec des noms de colonnes et de lignes pour temp pour permettre une meilleure lecture     
    total = pd.DataFrame(temp, columns = ['max iteration','recall','f1','accuracy'])
    print("\nscore total")
    print(total)
    
    print("Le meileur max iteration est : ")
    print(total.loc[total['accuracy'].idxmax(), 'max iteration'])
    
    #On crée un graphique pour représenter l'évolution du score accuracy en fonction du max d'itérations
    absisse = total.loc[:, ['max iteration']]
    ordonnee= total.loc[:, ['accuracy']]
    plt.title("Evolution du score accuracy en fonction du max d iteration")
    plt.plot(absisse,ordonnee)  # on crée la courbe
    plt.ylabel('score accuracy')
    plt.xlabel("max iteration")
    



   





