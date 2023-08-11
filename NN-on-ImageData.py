from tensorflow import keras
import numpy as np

input = np.array([0,0,0,1,1,0,0,0]) #learning the  array to NN
input

input = input.reshape(1, 8,1) #convert the array to culmnar array
input

keras.layers.InputLayer(input_shape =(8,1)), 
  keras.layers.Conv1D(1, 3, strides =1, activation= 'relu', name = 'layer1')  
])
model.summary()
model.get_layer('layer1').weights
mykernel = [np.array([[[0]],[[1]],[[0]]]), np.array([0])]
model.get_layer('layer1').set_weights(mykernel)
model.get_layer('layer1').weights
model.predict(input)
data = np.array(
    [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
        )

data.shape

data = data.reshape(1, 8,8, 1)

model2 = keras.Sequential([ 
  keras.layers.InputLayer(input_shape =(8, 8,1)),
  keras.layers.Conv2D(1, (3,3), strides = 1, activation= 'relu', name = 'layer2'),
  keras.layers.GlobalAvgPool2D()
])

model2.summary()

model2.get_layer('layer2').weights

mykernel2 = np.array([[[[0]], [[1]], [[0]]],
                      [[[0]], [[1]], [[1]]],
                      [[[1]], [[1]], [[0]]]])
bias= np.array([0])

model2.get_layer('layer2').set_weights([mykernel2, bias]) 
model2.get_layer('layer2').weights
model2.predict(data)

