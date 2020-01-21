# -*- coding: utf-8 -*-

#build the model
#%%
from keras.models import Sequential
from keras import layers
import keras
#%%
def build_model():
    #Instantiate an empty model
    model = Sequential()
    
    # C1 Convolutional Layer
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(50,50,3), padding='same'))
    
    # S2 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    
    # S4 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    # C5 Fully Connected Convolutional Layer
    model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    #Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())
    
    # FC6 Fully Connected Layer
    model.add(layers.Dense(84, activation='tanh'))
    
    #Output Layer with softmax activation
    model.add(layers.Dense(196, activation='softmax'))
    
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
    
    return model


#%%
if __name__ == '__main__':
    model=build_model()
    model.summary()


