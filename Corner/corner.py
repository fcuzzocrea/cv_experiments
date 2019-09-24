#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:55:59 2019

@author: fcuzzocrea
"""

from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters
import harris

# Importo l'immagine come array
im = array(Image.open('dumbo_cl.jpg').convert('L'))

print("Immagine Originale")
# Plotto i contorni
figure()
gray()
contour(im, origin='image')
axis('equal')
axis('off')
show()

print("Istogramma Immagine Originale")
# Plotto l'istogramma
figure()
hist(im.flatten(),128)  # flatten per ridurlo ad array monodimensionale
show()

# Inverto i livelli di grigio dell'immagine
im2 = 255 - im

# Restringo le intensita allintervallo 100 - 200
im3 = (100.0/255) * im + 100

# Qui invece applico una funzione quadratica per abbassare i valori dei pixel
# più scuri
im4 = 255.0 * (im/255.0)**2

print ((im.min()),(im.max()))
print ((im2.min()),(im2.max()))
print (int(im3.min()),int(im3.max()))
print (int(im4.min()),int(im4.max()))

# Ho convertito l'immagine di base in un array, ci ho applicato delle
# trasformazioni e dunque a questo punto devo rifarle diventare delle immagini
# da visualizzare
im2_t = Image.fromarray(uint8(im2))
im3_t = Image.fromarray(uint8(im3)) 
im4_t = Image.fromarray(uint8(im4))

# Plotto i contorni
print("Immagine invertita")
figure()
gray()
contour(im2, origin='image')
axis('equal')
axis('off')
show()

# Plotto i contorni
print("Immagine con livelli di intesita ristretti all'intervallo 100-200")
figure()
gray()
contour(im3, origin='image')
axis('equal')
axis('off')
show()

# Plotto i contorni
print("Immagine a cui ho applicato funzione per diminuire i valori dei pixel scuri")
figure()
gray()
contour(im4, origin='image')
axis('equal')
axis('off')
show()

# HISTOGRAM EQUALIZATION

nbr_bins = 256
imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)

 # Cumulative distribution function normalizzata 
cdf = imhist.cumsum() 
cdf = 255 * cdf / cdf[-1]

# A questo punto usiamo una interpolazione lineare della CDF per trovare
# i nuovi valori dei pixel
im5 = interp(im.flatten(),bins[:-1],cdf)
im5 = im5.reshape(im.shape)

# Plotto i contorni
print("Immagine a cui ho applicato funzione per diminuire i valori dei pixel scuri")
figure()
gray()
contour(im5, origin='image')
axis('equal')
axis('off')
show()

# Plotto l'istogramma
figure()
hist(im5.flatten(),128)  # flatten per ridurlo ad array monodimensionale
show()

# DERIVATA IMMAGINE 

# Facciamo la derivata usando i filtri di Sobel
im_x = zeros(im.shape)
filters.sobel(im,1,im_x)

im_y = zeros(im.shape)
filters.sobel(im,0,im_y)

magnitude = sqrt(im_x**2+im_y**2)

# Plotto i contorni
print("Derivata lungo x, Sobel")
figure()
gray()
contour(im_x, origin='image')
axis('equal')
axis('off')
show()

# Plotto i contorni
print("Derivata lungo y, Sobel")
figure()
gray()
contour(im_y, origin='image')
axis('equal')
axis('off')
show()

# Plotto i contorni
print("Modulo del gradiente, Sobel")
figure()
gray()
contour(magnitude, origin='image')
axis('equal')
axis('off')
show()

# Facciamo la derivata usando i filtri di Gauss (smoother)
sigma = 5
im_xg = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), im_xg)

im_yg = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (1,0), im_yg)

magnitude_g = sqrt(im_xg**2+im_yg**2)

# Plotto i contorni
print("Derivata lungo x, Gauss")
figure()
gray()
contour(im_xg, origin='image')
axis('equal')
axis('off')
show()

# Plotto i contorni
print("Derivata lungo y, Gauss")
figure()
gray()
contour(im_yg, origin='image')
axis('equal')
axis('off')
show()

# Plotto i contorni
print("Modulo, Gauss")
figure()
gray()
contour(magnitude_g, origin='image')
axis('equal')
axis('off')
show()

# CORNER DETECTION
sigma = 5
min_dist = 3
treshold = 0.05
harrisim = harris.compute_harris_response(im,sigma)

figure()
gray()
contour(harrisim, origin='image')
axis('equal')
axis('off')
show()

filtered_coords = harris.get_harris_points(harrisim, min_dist, treshold)
harris.plot_harris_points(im,filtered_coords)
