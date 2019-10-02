from_size=1000
to_size=30000
n_sample=10
n_restart=5

#####################################################################################################################################
# Dirichlet distribution
# CPC
python structural_scores.py --method=cpc --distribution=dirichlet --structure=asia --mode=multi --parameters 5 0.05 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
python plot_results.py --method=cpc --distribution=dirichlet --structure=asia --mode=multi --parameters 5 0.05 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart

# Elidan
python structural_scores.py --method=elidan --distribution=dirichlet --structure=asia --mode=multi --parameters 4 4\
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
python plot_results.py --method=elidan --distribution=dirichlet --structure=asia --mode=multi --parameters 4 4 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart
#####################################################################################################################################



#####################################################################################################################################
# Gaussian distribution
# CPC
python structural_scores.py --method=cpc --distribution=gaussian --correlation=0.8 --structure=asia --mode=multi --parameters 5 0.05 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
python plot_results.py --method=cpc --distribution=gaussian --correlation=0.8 --structure=asia --mode=multi --parameters 5 0.05 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart

# Elidan
python structural_scores.py --method=elidan --distribution=gaussian --correlation=0.8 --structure=asia --mode=multi --parameters 4 4\
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
python plot_results.py --method=elidan --distribution=gaussian --correlation=0.8 --structure=asia --mode=multi --parameters 4 4 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart
#####################################################################################################################################



#####################################################################################################################################
# Student distribution
# CPC
python structural_scores.py --method=cpc --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 5 0.05 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
python plot_results.py --method=cpc --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 5 0.05 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart

# Elidan
python structural_scores.py --method=elidan --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 4 4\
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
python plot_results.py --method=elidan --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 4 4 \
                            --from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart
#####################################################################################################################################
