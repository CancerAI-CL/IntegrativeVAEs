from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda
from keras.models import Model
import os
from cb_code.models.common import sse, bce, sampling


class HVAE:
    def __init__(self, args, type):
        self.args = args
        self.type = type
        self.vae = None
        self.encoder = None

    def build_model(self):
        if self.type == 'CNA':
            self.build_cna()
        elif self.type == 'RNA':
            self.build_rna()
        elif self.type == 'H':
            self.build_merged()
        else:
            raise ValueError('Unrecognised HVAE network type')

        # Define the loss
        kl_loss = 1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(self.reconstruction_loss + 1.0 * kl_loss)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
        self.vae.compile(optimizer=adam)
        self.vae.summary()

    def build_cna(self):
        # Build the encoder network
        # ------------ Input -----------------
        inp = Input(shape=(self.args.cna_input_size,))

        # ------------ Concat Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inp)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        self.z_mean = Dense(self.args.ds // 2, name='z_mean')(x)
        self.z_log_sigma = Dense(self.args.ds // 2, name='z_log_sigma')(x)
        z = Lambda(sampling, output_shape=(self.args.ds // 2,), name='z')([self.z_mean, self.z_log_sigma])

        self.encoder = Model(inp, [self.z_mean, self.z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ds // 2,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)

        # ------------ Out -----------------------
        cna_out = Dense(self.args.cna_input_size, )(x)

        decoder = Model(latent_inputs, cna_out, name='decoder')
        decoder.summary()

        output = decoder(self.encoder(inp)[2])
        self.vae = Model(inp, output, name='vae_cna')
        self.reconstruction_loss = bce(inp, output)

    def build_rna(self):
        # Build the encoder network
        # ------------ Input -----------------
        inp = Input(shape=(self.args.rna_input_size,))

        # ------------ Concat Layer -----------------
        x = Dense(self.args.ds, activation=self.args.act)(inp)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        self.z_mean = Dense(self.args.ds // 2, name='z_mean')(x)
        self.z_log_sigma = Dense(self.args.ds // 2, name='z_log_sigma')(x)
        z = Lambda(sampling, output_shape=(self.args.ds // 2,), name='z')([self.z_mean, self.z_log_sigma])

        self.encoder = Model(inp, [self.z_mean, self.z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ds // 2,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds, activation=self.args.act)(x)
        x = BN()(x)

        # ------------ Out -----------------------
        cna_out = Dense(self.args.rna_input_size, )(x)

        decoder = Model(latent_inputs, cna_out, name='decoder')
        decoder.summary()

        output = decoder(self.encoder(inp)[2])
        self.vae = Model(inp, output, name='vae_rna')
        self.reconstruction_loss = sse(inp, output)

    def build_merged(self):
        # Build the encoder network
        # ------------ Input -----------------
        inp = Input(shape=(self.args.ds,))

        # ------------ Concat Layer -----------------
        x = Dense(self.args.ds // 2, activation=self.args.act)(inp)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        self.z_mean = Dense(self.args.ls, name='z_mean')(x)
        self.z_log_sigma = Dense(self.args.ls, name='z_log_sigma')(x)
        z = Lambda(sampling, output_shape=(self.args.ls,), name='z')([self.z_mean, self.z_log_sigma])

        self.encoder = Model(inp, [self.z_mean, self.z_log_sigma, z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds // 2, activation=self.args.act)(x)
        x = BN()(x)

        # ------------ Out -----------------------
        cna_out = Dense(self.args.ds, )(x)

        decoder = Model(latent_inputs, cna_out, name='decoder')
        decoder.summary()

        output = decoder(self.encoder(inp)[2])
        self.vae = Model(inp, output, name='vae_merged')
        self.reconstruction_loss = sse(inp, output)

    def train(self, train, test):
        self.vae.fit(train, epochs=self.args.epochs, batch_size=self.args.bs, shuffle=True,
                     validation_data=(test, None))
        if self.args.save_model:
            self.vae.save_weights('./models/vae_x_mlp.h5')

    def predict(self, inp):
        return self.encoder.predict(inp, batch_size=self.args.bs)[0]
