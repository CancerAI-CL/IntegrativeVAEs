#!/bin/bash
for integ in 'Clin+mRNA' 'CNA+mRNA' 'Clin+CNA'
do
    for ds in 128 256 512
    do
        for lsize in 16 32 64
        do
            for distance in 'kl' 'mmd'
            do
                for beta in 1 10 15 25 50 100 
                do
                    for dtype in  'ER' 'DR' 'IC' 'PAM' #'W' whole data 
                    do
                        for fold in 1 2 3 4 5 #0 #whole data
                        do
                            python run_cncvae.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
                        done
                    done
                done
            done
        done
    done
done
