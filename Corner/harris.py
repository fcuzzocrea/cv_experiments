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
    
