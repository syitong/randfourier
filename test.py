import numpy as np
import matplotlib.pyplot as plt
import datagen, dataplot
import rff

X,Y = datagen.unit_circle_ideal(0.1,0.8,50)
dataplot.plot_circle(X,Y)
