import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from numpy import genfromtxt
from numpy import loadtxt
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
from keras.models import load_model
import sys
import keras
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Build the VGG16 network with ImageNet weights
model = load_model('mymodel.h5')
print('Model loaded.')

data = genfromtxt(sys.argv[1],delimiter=',',dtype=None)  
data = np.delete(data,0,0)                                   
num = data.shape[0]
feature = [ [int(x) for x in data[r,1].split()] for r in range (num)]
feature = np.array(feature)
#feature =feature/255
feature = feature.reshape((num,48,48,1))
feature = feature[:100,:,:,:]
out = model.predict(feature)
heatmaps = []
arr = [11,16,22,26]
for i in arr:
    pred_class = np.argmax(out[i])
    heatmap = visualize_saliency(model, 10 , [pred_class], feature[i])#10
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()