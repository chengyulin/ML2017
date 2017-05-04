from keras import applications
from keras.models import load_model
import numpy as np
from keras import backend as K
from scipy.misc import imsave
import matplotlib.pyplot as plt

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

model = load_model('mymodel.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'conv2d_4'

fig = plt.figure()
for  filter_index in range(10,30,2):
	
	input_img = model.input
	# build a loss function that maximizes the activation
	input_img_data = np.random.random((1, 48, 48 ,1 )) * 20 + 120

	layer_output = layer_dict[layer_name].output

	loss = K.mean(layer_output[:, :, :, filter_index])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img, K.learning_phase()], [loss, grads])

	# run gradient ascent for 20 steps
	step = 1
	for i in range(30):
	    loss_value, grads_value = iterate([input_img_data, False])
	    input_img_data += grads_value * step

	img = input_img_data[0]
	img = deprocess_image(img)
	tmp = fig.add_subplot(2,5,(filter_index-10)/2+1)
	tmp.imshow(img[:,0,:],cmap='BuGn')
plt.show()
# imsave('%s_filter_%d.png' % (layer_name, filter_index), img)