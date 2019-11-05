#!/bin/bash
for integ in 'Clin+mRNA' 'CNA+mRNA' 'Clin+CNA'
do
    for ds in 256
    do
        for lsize in 64
        do
            for distance in 'mmd'
            do
                for beta in 50 
                do
                    for dtype in  'W' #whole data 
                    do
                        for fold in 0 #whole data
                        do
                            python run_cncvae.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
                            python run_xvae.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
                            python run_mmvae.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
                            python run_hvae.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results'
                        done
                    done
                done
            done
        done
    done
done
