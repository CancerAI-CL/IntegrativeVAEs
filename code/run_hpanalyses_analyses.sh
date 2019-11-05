for integ in 'Clin+CNA'
do
    for model in 'CNCVAE' 'XVAE' 'MMVAE' 'HVAE'
    do
        python analyse_representations.py --integration=${integ} --model=${model} --dtype='ER' --numfolds=5 --resdir='results' --writedir='HParameters_Analyses'
    done
done


