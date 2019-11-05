import numpy as np
import argparse
import os

from models.xvae import XVAE
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding


configs = {
    128: {
 
        'ds': 128,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2

    },
    256: {
 
        'ds': 256,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },
    512: {

        'ds': 512,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA', type=str, required=True, default='Clin+mRNA')
parser.add_argument('--ds', help='The intermediate dense layers size', type=int, required=True)
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=str, default='0')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='ER')
parser.add_argument('--beta', help='beta size', type=int, required=True)
parser.add_argument('--distance', help='regularization', type=str, default='kl')
parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')



if __name__ == "__main__":
    args = parser.parse_args()
    config = configs[args.ds]
    for key, val in config.items():
        setattr(args, key, val)
 
    
if (args.fold == '0'): # whole data set
    
    print('TRAINING on the complete data')
    
    dataset = DatasetWhole('W')

    if (args.integration == 'Clin+mRNA'): #integrate Clin+mRNA
        s1_train = dataset.train['clin'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]

        
        
    elif (args.integration == 'Clin+CNA'): #integrate Clin+CNA
        s1_train = dataset.train['clin']
        s2_train = dataset.train['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]

        
    else:#integrate CNA+mRNA
        s1_train = dataset.train['cnanp'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])    
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]



    xvae = XVAE(args)
    xvae.build_model()


    xvae.train(s1_train, s2_train, s1_train, s2_train)
    emb_train = xvae.predict(s1_train, s2_train)

    if (args.writedir == ''):
        emb_save_dir = 'results/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    else:
        emb_save_dir = args.writedir+'/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    emb_save_file = args.dtype +'.npz'
    save_embedding(emb_save_dir,emb_save_file,emb_train)
    
else:

    print('TRAINING on the fold '+ format(args.fold))
    
    dataset = Dataset(args.dtype, args.fold)

    if (args.integration == 'Clin+mRNA'): #integrate Clin+mRNA
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train, s2_test = normalizeRNA(dataset.train['rnanp'],dataset.test['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]


        
    elif (args.integration == 'Clin+CNA'): #integrate Clin+CNA
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin']         
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]

    
    else:#integrate CNA+mRNA
        s1_train = dataset.train['cnanp']
        s1_test = dataset.test['cnanp']  
        s2_train, s2_test = normalizeRNA(dataset.train['rnanp'],dataset.test['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]



    xvae = XVAE(args)
    xvae.build_model()

  
    xvae.train(s1_train, s2_train, s1_test, s2_test)
    emb_train = xvae.predict(s1_train, s2_train)
    emb_test = xvae.predict(s1_test, s2_test)
    
    if (args.writedir == ''):
        emb_save_dir = 'results/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    else:
        emb_save_dir = args.writedir+'/XVAE_'+format(args.integration)+'_integration/xvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    emb_save_file = args.dtype +args.fold+'.npz'
    save_embedding(emb_save_dir,emb_save_file,emb_train, emb_test)
    
    