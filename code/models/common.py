from keras import backend as K
import tensorflow as tf

def sse(true, pred):
    return K.sum(K.square(true - pred), axis=1)


def cce(true, pred):
    return K.mean(K.sparse_categorical_crossentropy(true, pred, from_logits=True), axis=1)


def bce(true, pred):
    return K.sum(K.binary_crossentropy(true, pred, from_logits=True), axis=1)

def compute_kernel(x, y):
    x_size = K.shape(x)[0] #K.shape(x)[0] #need to fix to get batch size
    y_size = K.shape(y)[0]
    dim = K.int_shape(x)[1] #K.get_shape(x)[1] #x.get_shape().as_list()[1]   
    tiled_x = K.tile(K.reshape(x,K.stack([x_size, 1, dim])), K.stack([1,y_size, 1]))
    tiled_y = K.tile(K.reshape(y,K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
    kernel_input = K.exp(-K.mean((tiled_x - tiled_y)**2, axis=2)) / K.cast(dim, tf.float32)
    return kernel_input

def mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2*K.mean(xy_kernel)
    return mmd
    
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def kl_regu(z_mean,z_log_sigma):
    #regularizer. this is the KL of q(z|x) given that the 
    #distribution is N(0,1) (or any known distribution)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss