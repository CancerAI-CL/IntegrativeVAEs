import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os

import argparse


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings
warnings.filterwarnings('ignore')



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'T', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','F', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

train_labels=[]
test_labels=[]

train_raw=[]
test_raw=[]

train_cna=[]
test_cna=[]

train_rna=[]
test_rna=[]

train_clin=[]
test_clin=[]


parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA. Default is Clin+mRNA.', type=str, required=True, default='Clin+mRNA')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER). Default is ER.', type=str, default='ER')
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
parser.add_argument('--resdir', help='/PATH/TO/EMBEDDINGS - Default is  results/', type=str, required=True)
parser.add_argument('--numfolds', help='number of folds of CV-analyses, 0 indicates whole data set. Default is 5', type=int, default=5)
parser.add_argument('--model', help='Can be either CNCVAE, XVAE, MMVAE, HVAE or BENCH. BENCH refers to benchmark, i.e. learning a classifier on raw data and PCA transformed data. Default is CNCVAE', type=str, default='CNCVAE')
parser.add_argument('--tSNE', help='generate a tSNE plot (True/False). Default is false. Warning: it is computationaly demanding, and only enabled for whole dataset analyses', type=str2bool, default=False)


parser.add_argument('--NB', help='train a Naive Bayes Classifier. Default True', type=str2bool, default=True)
parser.add_argument('--SVM', help='train a SVM Classifier. Default False', type=str2bool, default=False)
parser.add_argument('--RF', help='train a Random Forest. Default False', type=str2bool, default=False)


if __name__ == "__main__":
    args = parser.parse_args()

 
 
    
if (args.dtype == 'W'):
    dataset = DatasetWhole('W')
    clin_train=dataset.train['clin']
    rna_train=normalizeRNA(dataset.train['rnanp'])
    cna_train=dataset.train['cnanp']
     
    train_clin.append(clin_train)
    train_rna.append(rna_train)
    train_cna.append(cna_train)
     
    if (args.integration=='Clin+mRNA'):
         train_raw.append(np.concatenate((clin_train,rna_train),axis=-1))
    
    elif (args.integration=='Clin+CNA'):
        train_raw.append(np.concatenate((clin_train,cna_train),axis=-1))
    else:
         train_raw.append(np.concatenate((cna_train,rna_train),axis=-1))

else:
 
    for fold in range(1,args.numfolds+1):
    
        dataset = Dataset(args.dtype,format(fold))
        clin_train=dataset.train['clin']
        clin_test=dataset.test['clin']
        rna_train, rna_test=normalizeRNA(dataset.train['rnanp'],dataset.test['rnanp'])
        cna_train=dataset.train['cnanp']
        cna_test=dataset.test['cnanp']
         
        train_clin.append(clin_train)
        train_rna.append(rna_train)
        train_cna.append(cna_train)
        
        test_clin.append(clin_test)
        test_rna.append(rna_test)
        test_cna.append(cna_test)
         
        if (args.integration=='Clin+mRNA'):
             train_raw.append(np.concatenate((clin_train,rna_train),axis=-1))
             test_raw.append(np.concatenate((clin_test,rna_test),axis=-1))
        elif (args.integration=='Clin+CNA'):
             train_raw.append(np.concatenate((clin_train,cna_train),axis=-1))
             test_raw.append(np.concatenate((clin_test,cna_test),axis=-1))
        else:
             train_raw.append(np.concatenate((cna_train,rna_train),axis=-1))
             test_raw.append(np.concatenate((cna_test,rna_test),axis=-1))
         
        if(args.dtype=='ER'):
             train_labels.append(dataset.train["ernp"])
             test_labels.append(dataset.test["ernp"])
        elif(args.dtype=='DR'):
            train_labels.append(dataset.train["drnp"])
            test_labels.append(dataset.test["drnp"])
        elif(args.dtype=='PAM'):
            train_labels.append(dataset.train["pam50np"])
            test_labels.append(dataset.test["pam50np"])
        elif(args.dtype=='IC'):
            train_labels.append(dataset.train["icnp"])
            test_labels.append(dataset.test["icnp"])
        else:   
            raise ValueError('Target not considered in this study')
         

if (args.resdir == ''):
        resdir = "results"
else:
        resdir=args.resdir
    


if (args.writedir == ''):
        savedir = "analyses"
else:
        savedir=args.writedir
if not os.path.exists(savedir):
    os.makedirs(savedir)    
        


if args.dtype=='W':
    for model in [args.model]:
        for dist in ['mmd']:
                for beta in [50]:
                    for ls in [64]:
                        for ds in [256]:
                            
                            
                            model_conf=format(model.lower())+'_LS_'+format(ls)+'_DS_'+format(ds)+'_'+format(dist)+'_beta_'+format(beta)
                            
                            if not os.path.isdir(resdir+'/'+model+'_'+format(args.integration)+'_integration/'+model_conf):
                                continue
                            else:
    
                                print("Generating tSNE plots...")
                                embed=np.load(resdir+'/'+model+'_'+format(args.integration)+'_integration/'+model_conf+'/'+format(args.dtype)+'.npz')
                                emb_train = embed['emb_train']
                                
                                pca= PCA(n_components=ls,random_state=42)
                                pca.fit(train_raw[0])#
                                pcaTrain=pca.transform(train_raw[0])
                                
                                tsneH = TSNE(random_state=42,perplexity=50)
                                tsne_embedH = tsneH.fit_transform(emb_train)
                                
                                tsneRAW = TSNE(random_state=42,perplexity=50)
                                tsne_raw = tsneRAW.fit_transform(train_raw[0])
                                
                                tsnePCA = TSNE(random_state=42,perplexity=50)
                                tsne_pca = tsnePCA.fit_transform(pcaTrain)
            
                                
                                
                                
                                
                                
                                
                                
                                fig=plt.figure(figsize=(18,8))
                                
                                plt.rcParams.update({'font.size': 22})
                                fig.suptitle (args.integration+" integration")
                                
                                plt.subplot(1, 3, 1)
                                plt.title('Raw data')
                                plt.xlabel('Latent dimension 1')
                                plt.ylabel('Latent dimension 2')
                                plt.scatter(tsne_raw[:, 0], tsne_raw[:, 1])
                                
                                plt.subplot(1, 3, 2)
                                plt.title('PCA')
                                plt.xlabel('Latent dimension 1')
                                plt.ylabel('Latent dimension 2')
                                plt.scatter(tsne_pca[:, 0], tsne_pca[:, 1])
                            
                                
                                plt.subplot(1, 3, 3)
                                plt.title(format(model))
                                plt.xlabel('Latent dimension 1')
                                plt.ylabel('Latent dimension 2')
                                plt.scatter(tsne_embedH[:, 0], tsne_embedH[:, 1])
                                
                                fig.savefig(savedir+'/tSNE_'+format(model_conf)+'.pdf',bbox_inches='tight')
                                fig.savefig(savedir+'/tSNE_'+format(model_conf)+'.jpg',bbox_inches='tight')
    
                                print('Done')
    
    
    
    
    

else:
    if args.model != 'BENCH':
        with open(savedir+'/'+format(args.model)+'_'+format(args.integration)+'_'+format(args.dtype)+'.csv', 'w') as f: 
            for model in [args.model]:
                print("------------------------------------------------",file=f)
                print(model,file=f)
                print("------------------------------------------------",file=f)
                print("model, type_integration, regularization, beta, latent_size, dense_layer_size, NB_Train_ACC, NB_Train_ACC_std, NB_Test_ACC, NB_Test_ACC_std, SVM_Train_ACC, SVM_Train_ACC_std, SVM_Test_ACC, SVM_Test_ACC_std, RF_Train_ACC, RF_Train_ACC_std, RF_Test_ACC, RF_Test_ACC_std ",file=f)
              
                
                for dist in ['mmd','kl']:
                    for beta in [1,10,15,25,50,100]:
                        for ls in [16,32,64]:
                            for ds in [128,256,512]:
                                
                    
                                    accsTrain_NB=[]
                                    accsTest_NB=[]
                                    
                                    accsTrain_SVM=[]
                                    accsTest_SVM=[]
                                    
                                    accsTrain_RF=[]
                                    accsTest_RF=[]
                                    
                                   
        
                                    note=""
                                    
                                    
                                    
                                    model_conf=model+'_'+format(args.integration)+'_integration/'+format(model.lower())+'_LS_'+format(ls)+'_DS_'+format(ds)+'_'+format(dist)+'_beta_'+format(beta)
                                   
                                    if not os.path.isdir(resdir+'/'+model_conf):
                                        continue
                                    else:
                                        print("Analysing: "+ model_conf)
                                        for fold in range(1,args.numfolds+1):
                                        
                                                                  
                                            embed=np.load(resdir+'/'+model_conf+'/'+format(args.dtype)+format(fold)+'.npz')
                                            emb_train = embed['emb_train']
                                            emb_test = embed['emb_test']
                                            
                                            random_state=42
                                            
                                            
            
                                            
                                            if (np.isnan(emb_train).any()):
                                               note+="Check *ER"+format(fold)+" embeding for problems" 
                                               print(resdir+'/'+model_conf+'/'+format(args.dtype)+format(fold)+'.npz is invalid. Consider re-training.' )
                                               continue
                                            else:
                                                  
                                                
                                                
                                                if args.NB:
                                                  nb=GaussianNB()
                                                  nb.fit(emb_train, train_labels[fold-1])
                                                
                                                  x_p_classes=nb.predict(emb_train)
                                                  accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                                                  accsTrain_NB.append(accTrain)
                                                   
                                                  y_p_classes=nb.predict(emb_test) 
                                                  accsTest_NB.append(accuracy_score(test_labels[fold-1],y_p_classes))
                                                else:
                                                  accsTrain_NB.append(0.0)
                                                  accsTest_NB.append(0.0)
                                                  
                                                if args.SVM:  
                                                  svm=SVC(C=1.5, kernel='rbf',random_state=42,gamma='auto')
                                                  svm.fit(emb_train, train_labels[fold-1])
                                                
                                                  x_p_classes=svm.predict(emb_train)
                                                  accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                                                  accsTrain_SVM.append(accTrain)
                                                   
                                                  y_p_classes=svm.predict(emb_test) 
                                                  accsTest_SVM.append(accuracy_score(test_labels[fold-1],y_p_classes))
                                                else:
                                                  accsTrain_SVM.append(0.0)
                                                  accsTest_SVM.append(0.0)
                                                  
                                                  
                                                if args.RF:
                                                  rf=RandomForestClassifier(n_estimators=50, random_state=42,  max_features=.5)
                                                  rf.fit(emb_train, train_labels[fold-1])
                                                
                                                  x_p_classes=rf.predict(emb_train)
                                                  accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                                                  accsTrain_RF.append(accTrain)
                                                   
                                                  y_p_classes=rf.predict(emb_test) 
                                                  accsTest_RF.append(accuracy_score(test_labels[fold-1],y_p_classes))
                                                else:
                                                  accsTrain_RF.append(0.0)
                                                  accsTest_RF.append(0.0)
           
           
                                    print(format(model)+','+format(args.integration)+','+format(dist)+','+format(beta)+','+format(ls)+','+format(ds)
                                    
                                                     +','+ format(np.mean(accsTrain_NB)) 
                                                    +','+ format(np.var(accsTrain_NB))
                                                    +','+ format(np.mean(accsTest_NB)) 
                                                    +','+ format(np.var(accsTest_NB))
                                                    
                                                    +','+ format(np.mean(accsTrain_SVM)) 
                                                    +','+ format(np.var(accsTrain_SVM))
                                                    +','+ format(np.mean(accsTest_SVM)) 
                                                    +','+ format(np.var(accsTest_SVM))
                                                    
                                                    +','+ format(np.mean(accsTrain_RF)) 
                                                    +','+ format(np.var(accsTrain_RF))
                                                    +','+ format(np.mean(accsTest_RF)) 
                                                    +','+ format(np.var(accsTest_RF))                            
                                                    +note,file=f)
                                    f.flush()
                                    print('Done.')
        f.close()    
    
    elif args.model == 'BENCH':      
                
        with open(savedir+'/benchmarks_'+format(args.integration)+'_'+format(args.dtype)+'.csv', 'w') as fb: 
            print("------------------------------------------------",file=fb)
            print("RawData",file=fb)
            print("------------------------------------------------",file=fb)
            print("model, type_integration, NB_Train_ACC, NB_Train_ACC_std, NB_Test_ACC, NB_Test_ACC_std, SVM_Train_ACC, SVM_Train_ACC_std, SVM_Test_ACC, SVM_Test_ACC_std, RF_Train_ACC, RF_Train_ACC_std, RF_Test_ACC, RF_Test_ACC_std ",file=fb)
            
            print('Analysing raw data...')
            
            accsTrain_NB=[]
            accsTest_NB=[]
            
            accsTrain_SVM=[]
            accsTest_SVM=[]
            
            accsTrain_RF=[]
            accsTest_RF=[]
            
            for fold in range(1,args.numfolds+1):
                if args.NB:
                  nb=GaussianNB()
                  nb.fit(train_raw[fold-1], train_labels[fold-1])
                
                  x_p_classes=nb.predict(train_raw[fold-1])
                  accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                  accsTrain_NB.append(accTrain)
                   
                  y_p_classes=nb.predict(test_raw[fold-1]) 
                  accsTest_NB.append(accuracy_score(test_labels[fold-1],y_p_classes))
                else:
                  accsTrain_NB.append(0)
                  accsTest_NB.append(0)
                  
                if args.SVM:  
                  svm=SVC(C=1.5, kernel='rbf',random_state=42,gamma='auto')
                  svm.fit(train_raw[fold-1], train_labels[fold-1])
                
                  x_p_classes=svm.predict(train_raw[fold-1])
                  accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                  accsTrain_SVM.append(accTrain)
                   
                  y_p_classes=svm.predict(test_raw[fold-1]) 
                  accsTest_SVM.append(accuracy_score(test_labels[fold-1],y_p_classes))
                else:
                  accsTrain_SVM.append(0)
                  accsTest_SVM.append(0)
                  
                  
                if args.RF:
                  rf=RandomForestClassifier(n_estimators=50, random_state=42,  max_features=.5)
                  rf.fit(train_raw[fold-1], train_labels[fold-1])
                
                  x_p_classes=rf.predict(train_raw[fold-1])
                  accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                  accsTrain_RF.append(accTrain)
                   
                  y_p_classes=rf.predict(test_raw[fold-1]) 
                  accsTest_RF.append(accuracy_score(test_labels[fold-1],y_p_classes))
                else:
                  accsTrain_RF.append(0)
                  accsTest_RF.append(0)
           
           
            print('raw data,'+format(args.integration)
        
                     +','+ format(np.mean(accsTrain_NB)) 
                    +','+ format(np.var(accsTrain_NB))
                    +','+ format(np.mean(accsTest_NB)) 
                    +','+ format(np.var(accsTest_NB))
                    
                    +','+ format(np.mean(accsTrain_SVM)) 
                    +','+ format(np.var(accsTrain_SVM))
                    +','+ format(np.mean(accsTest_SVM)) 
                    +','+ format(np.var(accsTest_SVM))
                    
                    +','+ format(np.mean(accsTrain_RF)) 
                    +','+ format(np.var(accsTrain_RF))
                    +','+ format(np.mean(accsTest_RF)) 
                    +','+ format(np.var(accsTest_RF))                            
                   ,file=fb)
            fb.flush()
            
            print('Done')
        
            accsTrain_NB=[]
            accsTest_NB=[]
            
            accsTrain_SVM=[]
            accsTest_SVM=[]
            
            accsTrain_RF=[]
            accsTest_RF=[]
            
            
            for comp in [16,32,64]:    
                print('Analysing PCA with '+format(comp)+ " components")
                for fold in range(1,args.numfolds+1):
                    
                    
                    pca=PCA(n_components=comp,random_state=42)
                    pca.fit(train_raw[fold-1])
                    pcaTrain=pca.transform(train_raw[fold-1])
                    pcaTest=pca.transform(test_raw[fold-1])
            
            
            
            
                    if args.NB:
                      nb=GaussianNB()
                      nb.fit(pcaTrain, train_labels[fold-1])
                    
                      x_p_classes=nb.predict(pcaTrain)
                      accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                      accsTrain_NB.append(accTrain)
                       
                      y_p_classes=nb.predict(pcaTest) 
                      accsTest_NB.append(accuracy_score(test_labels[fold-1],y_p_classes))
                    else:
                      accsTrain_NB.append(0)
                      accsTest_NB.append(0)
                      
                    if args.SVM:  
                      svm=SVC(C=1.5, kernel='rbf',random_state=42,gamma='auto')
                      svm.fit(pcaTrain, train_labels[fold-1])
                    
                      x_p_classes=svm.predict(pcaTrain)
                      accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                      accsTrain_SVM.append(accTrain)
                       
                      y_p_classes=svm.predict(pcaTest) 
                      accsTest_SVM.append(accuracy_score(test_labels[fold-1],y_p_classes))
                    else:
                      accsTrain_SVM.append(0)
                      accsTest_SVM.append(0)
                      
                      
                    if args.RF:
                      rf=RandomForestClassifier(n_estimators=50, random_state=42,  max_features=.5)
                      rf.fit(pcaTrain, train_labels[fold-1])
                    
                      x_p_classes=rf.predict(pcaTrain)
                      accTrain=accuracy_score(train_labels[fold-1],x_p_classes)
                      accsTrain_RF.append(accTrain)
                       
                      y_p_classes=rf.predict(pcaTest) 
                      accsTest_RF.append(accuracy_score(test_labels[fold-1],y_p_classes))
                    else:
                      accsTrain_RF.append(0)
                      accsTest_RF.append(0)
               
                print("Done")
                print('PCA_'+format(comp)+','+format(args.integration)
            
                         +','+ format(np.mean(accsTrain_NB)) 
                        +','+ format(np.var(accsTrain_NB))
                        +','+ format(np.mean(accsTest_NB)) 
                        +','+ format(np.var(accsTest_NB))
                        
                        +','+ format(np.mean(accsTrain_SVM)) 
                        +','+ format(np.var(accsTrain_SVM))
                        +','+ format(np.mean(accsTest_SVM)) 
                        +','+ format(np.var(accsTest_SVM))
                        
                        +','+ format(np.mean(accsTrain_SVM)) 
                        +','+ format(np.var(accsTrain_SVM))
                        +','+ format(np.mean(accsTest_SVM)) 
                        +','+ format(np.var(accsTest_SVM))                            
                        ,file=fb)
                fb.flush()
        fb.close()
    else:
        raise ValueError("Not supported model: "+ args.model)

        
