#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:42:47 2019

@author: fcuzzocrea
"""

from scipy.ndimage import filters
from numpy import *
from pylab import *
from scipy.ndimage import filters

def compute_harris_response(im,sigma):
    """ Compute the Harris Corner response function for each pixel in a graylevel image """
    
    # Derivate
    imx = zeros(im.shape)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    # Calcolo la matrice di Harris (è un hessiana)
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    
    # Cacolo determinante e traccia delle matrice di Harris per calcolarmi 
    # la response function
    W_det = Wxx*Wyy - Wxy**2
    W_tr = Wxx + Wyy
    R = W_det / W_tr
    
    return R

def get_harris_points(harrisim, min_dist, treshold):
    """
        Return corners from an Harris reponse image 
    
        [min_dist] is the minumum number of pixel separating corners and image boundary 
    """
    
    # Cerchiamo i punti candidati ad essere dei "corner" andando a selezionare
    # quelli che sono sopra una certa treshold
    corner_threshold = harrisim.max() * treshold
    harrisim_t = (harrisim > corner_threshold) * 1 # Da True, False a 0,1
    
    # Ricavo le coordinate dei punti candidati
    # Sono quelli diversi da zero, li piglio e li traspongo
    coords = array(harrisim_t.nonzero()).T
    
    # Itera per estrarre i candidati corner dalla HRF (harris response fctn)
    candidate_values = zeros(coords.shape[0])
    for c in range(0,coords.shape[0]) :
       candidate_values[c] = harrisim[coords[c,0],coords[c,1]]
       
    # Alternativamente puoi fare il conto in una linea e salvare i risultati
    # in una lista cosi :
    #candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    # Li ordiniamo in ordine discendente di HRF
    index = argsort(candidate_values)
    
    # Conserviamo i punti ammessi in un vettore (prepariamo solo la memoria)
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # Selezionamo i punti migliori tenndo anche conto della questione della
    # distanza minima, cacciando fuori i punti selezionati come angoli ma che 
    # sono vicini gia ad altri angoli
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1: # Mi pesca quelli = a 1
            filtered_coords.append(coords[i]) # aggiungi all'array
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
            (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
                              
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    """ Plots corners found in image. """
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords],'*')
    axis('off')
    show()
    
def get_descriptors(image, filtered_coords, width):
    """ For each point return the pixel values around the point using a 
        window of width 2*width+1 (the points are extracted with such as the
        minimum distance between two points is greater that width) """
        
    # Vettore vuoto per allocare la memoria, conterra le varie "finestre" centrate
    # nei punti di interessa che ho estrattto con harris
    desc = []
    
    # Per ogni punto di interesse adesso mi estraggo una finestra centrata nel
    # punto di interesse, e mi conservo tutte le finestre nella lista desc
    # (E' una lista, non è un vettore.)
    for coords in filtered_coords:
        patch = image[coords[0] - width:coords[0] + width + 1, coords[1] - width : coords[1] + width + 1].flatten()
        # Aggiungo la finestra alla lista dei descrittori
        desc.append(patch)
    
    return desc

def match(desc1, desc2, treshold):
    """ For each corner point descriptor in the first image, select its match
        to the second image using norma normalized cross-correlation """
        
    # OK, adesso dobbiamo matchare ciò che c'è nella prima immagine a ciò che
    # c'è nella seconda immagine, e lo facciamo tramite la normalized
    # cross-correlation function
    
    # Numero di pixel nella finestra
    n = len(desc1[0])
    
    # Prealloco la memoria, il
    d = -ones((len(desc1),len(desc2))) 
    
    # Mi calcolo la normalized cross correlation function per ogni finestra
    # centrata nel punto di interesse
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            I1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            I2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc = sum(I1*I2) / (n - 1)
            # Qui cerchiamo di non buttare dentro punti troppo vicini
            if ncc > treshold:
                d[i,j] = ncc      # altrimenti resta -1
    
    # Argsort ritorna gli indici che mi sortano l'array in ordine crescente            
    ndx = argsort(-d)
    
    # Qui si estrapola gli indici della prima colonna sortati
    matchscores = ndx[:,0]
    
    return matchscores

# Adesso lui implemente una funzione che fa il match al contrario (dalla 
# seconda alla prima) cosi si caccia fuori i punti che hanno un buon match  
# in tutte e due le direzioni
    
def match_twosided(desc1,desc2,treshold):
    """ Two sided symmetric version of match """
    
    # Applico semplicemente le funzioni sopra definite
    matches_12 = match(desc1, desc2, treshold)
    matches_21 = match(desc2, desc1, treshold)
    
    # Pesca l'elemento 0 da where che gli viene fuori da where, questi sono i
    # punti  che non matchano
    ndx_12 = where(matches_12 <=0)[0]
    
    # tQuindi togliamo i match non simmetrici sostituendoli con -1
    for n in ndx_12:
        if matches_21[matches_12[n]] !=n:
            matches_12[n] = -1
            
    return matches_12

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    
    # Prepara la memoria
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
     # Seleziona le immagini aventi meno colonne e riempie le colonne mancanti 
     # con gli zeri (cosi che le due immagini abbiano un numero eguale di 
     # colonne)    
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
 
     # Ovviamente se nessuno di questi due casi si verifica allora rows1 = rows2
     # e non è necessario alcun riempimento    
    return concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches 
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations), 
        matchscores (as output from 'match()'), 
        show_below (if images should be shown below matches). """
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    
    imshow(im3)
    
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')
    