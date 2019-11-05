for integ in 'Clin+mRNA' 'CNA+mRNA' 'Clin+CNA'
do
   python analyse_representations.py --integration=${integ} --model='HVAE' --dtype='W' --resdir='results' --writedir='tsne_Analyses'
done


