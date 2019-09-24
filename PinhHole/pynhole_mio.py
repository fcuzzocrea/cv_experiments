#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:23:45 2019

@author: fcuzzocrea
"""

from numpy import *
from scipy import linalg
import matplotlib.pyplot as plt

# Codice preso da SO per plottare in una nuova finestra invece che inline
# (alla matlab)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

""" 
    Il modello di camera pinole mi dice che posso scrivere la connessione 
    che c'è fra un generico punto 3D M (piano pi greca) ed il punto proiettato 
    sul piano dell'immagine come : 
        
                        m = dot(P,M)
                        
    Dove P = K [R' -R't]   -----> MATRICE 3X4 (vedi appunti grimorio)
"""

# Inizializziamo i parametri
# hstack() è una funzione comoda che mi permette di aggiungere un vettore
# orizzontale ad un altro vettore o ad una matrice (come in quest caso)
# Per adesso la matrice P è composta solo da K e t, e stiamo supponendo che
# le lunghezze focali x e y siano 1. Il -10 non so cosa significhi  
P = hstack((eye(3),array([[0],[0],[1]])))

# Per adesso gli altri parametri non ci interessano
K = 0
R = 0
t = 0

# Importo il modello 3D (.T lo traspone per fare la moltiplicazione)
M = loadtxt('house.p3d').T

# Con vstack aggiungiamo una riga di 1 alla fine
# shape serve per le dimensioni. shape.[1] mi seleziona le colonne (672)
M = vstack((M,ones(M.shape[1])))

# A questo punto ci calcoliamo la proiezione nel piano dell'immagine
# (m) dei punti reali 3D (M)
m = dot(P,M) 

# Plottiamo l'immagine proiettata con il modello di camera pinhole
plt.figure()
plt.plot(m[0],m[1],'k.')
plt.show()
