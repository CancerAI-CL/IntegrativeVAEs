from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda, Dropout
from keras.models import Model

import numpy as np


from models.common import sse, bce, mmd, sampling, kl_regu
from keras.losses import mean_squared_error,binary_crossentropy

from tensorflow import set_random_seed


class MMVAE:
    def __init__(self, args):
        self.args = args
        self.vae = None
        self.encoder = None

    def build_model(self):
        np.random.seed(42)
        set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        s1_inp = Input(shape=(self.args.s1_input_size,))
        s2_inp = Input(shape=(self.args.s2_input_size,))
        inputs = [s1_inp, s2_inp]

        # ------------ Encoding Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(s1_inp)
        x = BN()(x)

        encoder_s1 = Dense(self.args.ds, activation=self.args.act)(s1_inp)
        encoder_s1 = BN()(encoder_s1)
        
        encoder_s2 = Dense(self.args.ds, activation=self.args.act)(s2_inp)
        encoder_s2 = BN()(encoder_s2)

        merged_L=Concatenate(axis=-1)([encoder_s1, encoder_s2])
        merged_R=Concatenate(axis=-1)([encoder_s2, encoder_s1])

        encoder_s1 = Dense(self.args.ds, activation=self.args.act)(merged_R)
        encoder_s1 = BN()(encoder_s1)
        encoder_s2 = Dense(self.args.ds, activation=self.args.act)(merged_L)
        encoder_s2 = BN()(encoder_s2)

        merged_layer=Concatenate(axis=-1)([encoder_s1, encoder_s2])

        encoder = Dense(self.args.ds, activation=self.args.act)(merged_layer)
        encoder = BN()(encoder)

      

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ls, name='z_mean')(encoder)
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(encoder)
        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        decoder = Dense(self.args.ds, activation=self.args.act)(x)
        decoder = BN()(decoder)

        
        decoder=Dropout(self.args.dropout)(decoder)

        decoder_s1 = Dense(self.args.ds, activation=self.args.act)(decoder)
        decoder_s1 = BN()(decoder_s1)


        decoder_s2 = Dense(self.args.ds, activation=self.args.act)(decoder)
        decoder_s2  = BN()(decoder_s2)


        outputs=[decoder_s1, decoder_s2]

        # ------------ Out -----------------------
        s1_out = Dense(self.args.s1_input_size, activation='sigmoid')(decoder_s1)
        
        if self.args.integration == 'Clin+CNA':
            s2_out = Dense(self.args.s2_input_size,activation='sigmoid')(decoder_s2)
        else:
            s2_out = Dense(self.args.s2_input_size)(decoder_s2)

        decoder = Model(latent_inputs, [s1_out, s2_out], name='decoder')
        decoder.summary()

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        # Define the loss
        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ls]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)


        s1_loss= binary_crossentropy(inputs[0], outputs[0])

        if self.args.integration == 'Clin+CNA':
            s2_loss =binary_crossentropy(inputs[1], outputs[1])
        else:
            s2_loss =mean_squared_error(inputs[1], outputs[1])
        
        
        
        reconstruction_loss = s1_loss+s2_loss

        vae_loss = K.mean(reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()

    def train(self, s1_train, s2_train, s1_test, s2_test):
        self.vae.fit([s1_train, s2_train], epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True,
                     validation_data=([s1_test, s2_test], None))
        if self.args.save_model:
            self.vae.save_weights('./models/vae_mmvae.h5')

    def predict(self, s1_data, s2_data):
        return self.encoder.predict([s1_data, s2_data], batch_size=self.args.bs)[0]
