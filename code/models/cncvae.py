from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from keras.models import Model

import tensorflow as tf
import numpy as np

import os
from models.common import sse, bce, mmd, sampling, kl_regu
from keras.losses import mean_squared_error,binary_crossentropy
import numpy as np



class CNCVAE:
    def __init__(self, args):
        self.args = args
        self.vae = None
        self.encoder = None

    def build_model(self):
        np.random.seed(42)
        tf.random.set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        inputs = Input(shape=(self.args.input_size,), name='concat_input')
        #inputs = [concat_inputs]

        # ------------ Encoding Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inputs)
        x = BN()(x)      

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ls, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)
        
        x=Dropout(self.args.dropout)(x)

        # ------------ Out -----------------------
        
        if self.args.integration == 'Clin+CNA':
            concat_out = Dense(self.args.input_size,activation='sigmoid')(x)
        else:
            concat_out = Dense(self.args.input_size)(x)
        
        decoder = Model(latent_inputs, concat_out, name='decoder')
        decoder.summary()

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        # Define the loss
        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ls]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)


        if self.args.integration == 'Clin+CNA':
            reconstruction_loss = binary_crossentropy(inputs, outputs)
        else:
            reconstruction_loss = mean_squared_error(inputs, outputs)
        vae_loss = K.mean(reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()

    def train(self, s1_train, s2_train, s1_test, s2_test):
        train=np.concatenate((s1_train,s2_train), axis=-1)
        test=np.concatenate((s1_test,s2_test), axis=-1)
        self.vae.fit(train, epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True, validation_data=(test, None))
        if self.args.save_model:
            self.vae.save_weights('./models/vae_cncvae.h5')

    def predict(self, s1_data, s2_data):

        return self.encoder.predict(np.concatenate((s1_data, s2_data), axis=1), batch_size=self.args.bs)[0]
