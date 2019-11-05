from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda, Dropout
from keras.models import Model

from models.common import sse, bce, mmd, sampling, kl_regu
import numpy as np
from tensorflow import set_random_seed
from keras.losses import mean_squared_error, binary_crossentropy


class HVAE:
    def __init__(self, args, type):
        self.args = args
        self.type = type
        self.vae = None
        self.encoder = None

    def build_model(self):
        if self.type == 'catVAE':
            self.build_s1()
        elif self.type == 'numVAE':
            self.build_s2()
        elif self.type == 'H':
            self.build_merged()
        else:
            raise ValueError('Unrecognised HVAE network type')




      


#        # Define the loss
#        kl_loss = 1 + z_log_sigma - K.square(self.z_mean) - K.exp(z_log_sigma)
#        kl_loss = K.sum(kl_loss, axis=-1)
#        kl_loss *= -0.5
#        vae_loss = K.mean(self.reconstruction_loss + self.args.beta * kl_loss)
#        self.vae.add_loss(vae_loss)


    def build_s1(self):
        np.random.seed(42)
        set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        inp = Input(shape=(self.args.s1_input_size,))

        # ------------ Concat Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inp)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ds // 2, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ds // 2, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ds // 2,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inp, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ds // 2,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)
        x=Dropout(self.args.dropout)(x)
        # ------------ Out -----------------------
        s1_out = Dense(self.args.s1_input_size, activation='sigmoid')(x)

        decoder = Model(latent_inputs, s1_out, name='decoder')
        decoder.summary()

        output = decoder(self.encoder(inp)[2])
        self.vae = Model(inp, output, name='vae_s1')
        self.reconstruction_loss = binary_crossentropy(inp, output)
                
     
        
          # Define the loss
        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ds // 2]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)

     
        vae_loss = K.mean(self.reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()

        
        
        

    def build_s2(self):
        np.random.seed(42)
        set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        inp = Input(shape=(self.args.s2_input_size,))

        # ------------ Concat Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inp)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ds // 2, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ds // 2, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ds // 2,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inp, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ds // 2,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)
        x=Dropout(self.args.dropout)(x)
        # ------------ Out -----------------------
        s2_out = Dense(self.args.s2_input_size, )(x)

        decoder = Model(latent_inputs, s2_out, name='decoder')
        decoder.summary()

        output = decoder(self.encoder(inp)[2])
        self.vae = Model(inp, output, name='vae_s2')
        self.reconstruction_loss = mean_squared_error(inp, output)
        
          # Define the loss
        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ds // 2]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)

     
        vae_loss = K.mean(self.reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()


    def build_merged(self):
        np.random.seed(42)
        set_random_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        inp = Input(shape=(self.args.ds,))

        # ------------ Concat Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inp)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        z_mean = Dense(self.args.ls, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inp, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)
        x=Dropout(self.args.dropout)(x)
        # ------------ Out -----------------------
        out = Dense(self.args.ds)(x)

        decoder = Model(latent_inputs, out, name='decoder')
        decoder.summary()

        output = decoder(self.encoder(inp)[2])
        self.vae = Model(inp, output, name='vae_merged')
        self.reconstruction_loss = mean_squared_error(inp, output)
        
          # Define the loss
        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ls]))
            distance = mmd(true_samples, z)
        if self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)

     
        vae_loss = K.mean(self.reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()


    def train(self, train, test):
        self.vae.fit(train, epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True,
                     validation_data=(test, None))
        if self.args.save_model:
            self.vae.save_weights('./models/vae_hvae_mlp.h5')

    def predict(self, inp):
        return self.encoder.predict(inp, batch_size=self.args.bs)[0]
