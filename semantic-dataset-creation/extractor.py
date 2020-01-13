import os
import ast
import numpy as np
from abc import ABC, abstractmethod

from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from dbn import *

class Extractor(ABC):
    """
    A model for extracting features from token vectors
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_features(self, input_vecs, label_vecs):
        pass


class ConvolutionalAutoencoder(Extractor):
    """ 
    A convolutional autoencoder for feature extraction
    """
    def get_features(self, input_vecs, label_vecs):
        self.input_vecs = input_vecs
        self.model, self.encoder = self.__init_model(vecs_shape=self.input_vecs.shape)
        self.input_vecs = self.input_vecs.reshape((self.input_vecs.shape[0], self.input_vecs.shape[1], 1))
        self.__train(self.input_vecs)
        representations = self.encoder.predict(self.input_vecs)
        return representations


    def __init_model(self, vecs_shape):
        print(vecs_shape)
        input_vec = Input(shape=(vecs_shape[1], 1))
        x = Convolution1D(16, 3, activation='relu', padding='same')(input_vec)
        x = MaxPooling1D(2, padding='same')(x)
        x = Convolution1D(8, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Convolution1D(8, 3, activation='relu', padding='same')(x)
        encoder = MaxPooling1D(2, padding='same')(x)

        y = Flatten()(x)
        y = Dense(100, activation='softmax')(y)

        x = Convolution1D(8, 3, activation='relu', padding='same')(encoder)
        x = UpSampling1D(2)(x)
        x = Convolution1D(8, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Convolution1D(16, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoder = Convolution1D(1, 3, activation='sigmoid', padding='same')(x)

        encoder_model = Model(input_vec, y)
        autoencoder = Model(input_vec, decoder)

        return autoencoder, encoder_model


    def __train(self, train):
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model.fit(train, train, epochs=1000, batch_size=20, shuffle=True)



class DeepAutoencoder(Extractor):
    """ 
    A deep autoencoder for feature extraction
    """
    def get_features(self, input_vecs, label_vecs):
        self.input_vecs = input_vecs
        self.model, self.encoder = self.__init_model(vecs_shape=self.input_vecs.shape)
        self.input_vecs = self.input_vecs.reshape((self.input_vecs.shape[0], self.input_vecs.shape[1]))
        self.__train(self.input_vecs)
        representations = self.encoder.predict(self.input_vecs)
        return representations


    def __init_model(self, vecs_shape):
        input_vec = Input(shape=(vecs_shape[1],))
        x = Dense(128, activation='relu')(input_vec)
        x = Dense(100, activation='relu')(x)
        encoder = Dense(100, activation='relu')(x)

        x = Dense(100, activation='relu')(encoder)
        x = Dense(128, activation='relu')(x)
        decoder = Dense(vecs_shape[1], activation='sigmoid')(x)

        encoder_model = Model(input_vec, encoder)
        autoencoder = Model(input_vec, decoder)
        return autoencoder, encoder_model


    def __train(self, train):
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model.fit(train, train, epochs=1000, batch_size=20, shuffle=True)



class DeepBeliefNetwork(Extractor):
    """ 
    A deep belief network for feature extraction
    """
    def get_features(self, input_vecs, label_vecs):
        self.input_vecs = Variable(torch.from_numpy(input_vecs)).type(torch.FloatTensor)
        self.label_vecs = Variable(torch.from_numpy(label_vecs)).type(torch.FloatTensor)
        self.visible_units = input_vecs.shape[1]
        self.hidden_units = [200, 100, 100]
        self.model = self.__init_model(self.visible_units, self.hidden_units)
        self.__train(self.input_vecs, self.label_vecs)
        representations = []
        for inst in self.input_vecs:
            reconstructed = inst.view(1, -1).type(torch.FloatTensor)
            _, reconstructed = self.model.reconstruct(reconstructed)
            representations.append(reconstructed)
        representations = np.array([t.numpy() for t in representations])
        return representations


    def __init_model(self, visible, hidden):
        model = DBN(visible_units=visible, hidden_units=hidden,
                        k = 5, learning_rate = 0.01, learning_rate_decay = True,
                        xavier_init = True, increase_to_cd_k = False, use_gpu = False)
        return model


    def __train(self, train, label):
        self.model.train_static(train, label, 1000, 20)




    
        


        

