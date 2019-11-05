from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from keras.models import Model

from models.common import sse, bce, mmd, sampling, kl_regu
from keras.losses import mean_squared_error,binary_crossentropy
import numpy as np
from tensorflow import set_random_seed

class XVAE:
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

        # ------------ Concat Layer -----------------
        x1 = Dense(self.args.ds, activation=self.args.act)(s1_inp)
        x1 = BN()(x1)

        x2 = Dense(self.args.ds, activation=self.args.act)(s2_inp)
        x2 = BN()(x2)

        x = Concatenate(axis=-1)([x1, x2])

        x = Dense(self.args.ds, activation=self.args.act)(x)
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
        # ------------ Dense branches ------------
        x1 = Dense(self.args.ds, activation=self.args.act)(x)
        x1 = BN()(x1)
        x2 = Dense(self.args.ds, activation=self.args.act)(x)
        x2 = BN()(x2)

        # ------------ Out -----------------------
        s1_out = Dense(self.args.s1_input_size, activation='sigmoid')(x1)
        
        if self.args.integration == 'Clin+CNA':
            s2_out = Dense(self.args.s2_input_size,activation='sigmoid')(x2)
        else:
            s2_out = Dense(self.args.s2_input_size)(x2)

        decoder = Model(latent_inputs, [s1_out, s2_out], name='decoder')
        decoder.summary()

        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_x')

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

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False, decay=0.001)
        self.vae.compile(optimizer=adam, metrics=[mean_squared_error, mean_squared_error])
        self.vae.summary()

    def train(self, s1_train, s2_train, s1_test, s2_test):
        self.vae.fit([s1_train, s2_train], epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True,
                     validation_data=([s1_test, s2_test], None))
        if self.args.save_model:
            self.vae.save_weights('./models/vae_xvae.h5')

    def predict(self, s1_data, s2_data):
        return self.encoder.predict([s1_data, s2_data], batch_size=self.args.bs)[0]
