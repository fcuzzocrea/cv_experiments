# PinchHole Camera Model

from numpy import *
from scipy import linalg
import matplotlib.pyplot as plt

class Camera (object):
    """ Class for representing pin-hole camera """
    
    def __init__(self, P):
        """ This initializes the Camera Parameter Matrix P = K[R'-R't]M used to model the camera """
        self.P = P
        self.K = None # Calibration Matrix
        self.R = None # Rotation Matrix
        self.t = None # Traslation Vector
        self.c = None # Camera Center
        
    def project(self, M):
        """ Project point towards M (4*n array) and normalize coordinates """
        
        m = dot(self.P,M)
        for i in range(3):
            m[i] = m[i] / m[2]
        return m
    
#import camera

# load points
points = loadtxt('house.p3d').T
points = vstack((points,ones(points.shape[1])))

# setup camera
P = hstack((eye(3),array([[0],[0],[-10]])))
cam = Camera(P)
x = cam.project(points)

# Plottiamo l'immagine proiettata con il modello di camera pinhole
plt.figure()
#figure()
plt.plot(x[0],x[1],'k.')
plt.show()


