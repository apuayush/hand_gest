import os
import cv2
import imutils
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import pairwise
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# keras modules
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop
from keras.utils.vis_utils import plot_model