for integ in 'Clin+mRNA' 'CNA+mRNA' 'Clin+CNA'
do
    for model in 'CNCVAE' 'XVAE' 'MMVAE' 'HVAE' 'BENCH'
    do
        for label in 'DR' 'PAM' 'IC'
        do
            python analyse_representations.py --integration=${integ} --model=${model} --dtype=${label} --numfolds=5 --resdir='results' --writedir='BestModel_Analyses' --NB='True' --SVM='True' --RF='True'
        done
    done
done


