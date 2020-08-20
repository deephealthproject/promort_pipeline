# Copyright (c) 2019-2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys,os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.backend import clear_session
    from tensorflow.keras.layers import Dense, Reshape
    clear_session()
    tf_gpu = True
    if tf.__version__ < "2.2.0":
        print ("Tensorflow version is %f. It must be >= 2.2")
        tf_gpu = False

    import keras2onnx
except:
    tf_gpu = False
    print ('Cannot import tensoflow')

## Load tensorflow model
if len(sys.argv) == 3:
    model_fn = sys.argv[1]
    weights_fn = sys.argv[2]
else:
    print("Usage: %s model_fn weight_fn" % sys.argv[0])
    sys.exit(-1)

json_file = open(model_fn, 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create model
model = model_from_json(loaded_model_json)

## Load weights 
model.load_weights(weights_fn)
print("Keras model loaded from disk:")
model.summary()

print ("\n\nConverting to onnx for eddl...")
print ("Step1: Changing data format to channel first... ", end = '')

### Convert the convolutional part of the network to the channel first data format
tf.keras.backend.set_image_data_format('channels_first')
new_shape = (3, 256, 256)
vgg16_old_model = model.layers[0]

vgg16_new_model = tf.keras.applications.VGG16(weights=None, input_shape=new_shape, include_top=False)
for new_layer, layer in zip(vgg16_new_model.layers[1:], vgg16_old_model.layers[1:]):
    new_layer.set_weights(layer.get_weights())

print ("Done")
print ("Step2: Replacing flatten layer with reshape layer before the FCN classifier... ", end = '')

new_model = tf.keras.Sequential()
new_model.add(vgg16_new_model)
new_model.add(Reshape((-1,)))  ## Flatten
new_model.add(model.layers[2]) ## dense layer
new_model.add(model.layers[3]) ## Out dense layer

print ("Done")
print ("The model to be converted to ONNX is:")
new_model.summary()

print ("Step3: Converting to ONNX and saving the model... ", end = '')
onnx_model = keras2onnx.convert_keras(new_model, new_model.name)
keras2onnx.save_model(onnx_model, 'keras-vgg16.onnx')
print ("Done")

