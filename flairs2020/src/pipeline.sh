#!/bin/bash

# One Script to rule them all, One Script to find them,
# One Script to bring them all and in the darkness bind them.

DATA_DIR_PREFIX="../data"
RES_DIR_PREFIX="../results"
FIG_DIR_PREFIX="../figures"

#compute="both"
distribution="gaussian"
correlation="0.8"
structure="asia"

mode="multi"
method="cpc"

from_size=1000
to_size=30000
n_sample=10
n_restart=5
sample_size=50000

mcss=5      # Maximum size for the conditioning set in continuous PC
alpha=0.05  # Confidence level for continuous PC

mp=4        # Maximum parent in Elidan learning
hcr=4       # Number of restart for the hill climbing in Elidan

#data_exist=false
#results_exist=false
#figures_exist=false

usage()
{
    echo "usage: pipeline [[[-D distrib] [-C corr] [-S struct]] | [-h]]"
}

# Checking which option(s) has been given
while [ "$1" != "" ]; do
    case $1 in
        -D | --distribution )
            shift
            distribution=$1;;
        -C | --correlation )
            shift
            correlation=$1;;
        -S | --structure )
            shift
            structure=$1;;
        --compute )
            shift
            compute=$1;;
        -h | --help )
            usage
            exit;;
        * )
            usage
            exit 1
    esac
    shift
done

DIR_SUFFIX="$distribution/$structure/r${correlation//./}"

DATA_DIR="$DATA_DIR_PREFIX/samples/$DIR_SUFFIX"
STRUCT_DIR="$DATA_DIR_PREFIX/structures"
RES_DIR="$RES_DIR_PREFIX/$DIR_SUFFIX"
FIG_DIR="$FIG_DIR_PREFIX/$DIR_SUFFIX"

FILE_NAME="scores_${mode}_${method}_${structure}_${distribution}"
FILE_NAME="${FILE_NAME}_f${from_size}t${to_size}s${n_sample}r${n_restart}" 
FILE_NAME="${FILE_NAME}mcss5alpha5"

if [ -f "$STRUCT_DIR/$structure.txt" ]; then
    echo "The file $structure.txt exists."
else
    echo "The file $structure.txt doesn't exist."
    exit 1
fi

if [ -d "$DATA_DIR" ]; then
    if [ $(ls -1 | wc -l) -lt $n_restart ]; then
        response=
        echo "There is less file than wanted number of restarts. Generate more files? (y/n) > "
        read response
        if [ "$response" != "y" ]; then
            echo "Exiting program."
            exit 1
        fi
        python generate_data.py --distribution=$distribution --structure=$structure \
                                --n_sample=$n_restart --sample_size=$sample_size \
                                --correlation=$correlation
    fi
else
    echo "No corresponding data have been found. Generating some..."
    python generate_data.py --distribution=$distribution --structure=$structure \
                            --n_sample=$n_restart --sample_size=$sample_size \
                            --correlation=$correlation
fi

if [ -f "$RES_DIR/$FILE_NAME.csv" ]; then
    echo "Result file exists."
else
    echo "Result file does'nt exist."
    echo "Doing scientific stuff to generate one..."
    python structural_scores.py --method=$method \
                                --distribution=$distribution \
                                --correlation=$correlation \
                                --structure=$structure \
                                --mode=multi \
                                --parameters 5 0.05 \
                                --from_size=$from_size \
                                --to_size=$to_size \
                                --n_sample=$n_sample \
                                --n_restart=$n_restart
fi


if [ -f "$FIG_DIR/$FILE_NAME.pdf" ]; then
    echo "Figure file exists."
else
    echo "Figure file does'nt exist."
    echo "Doing scientific stuff to generate one..."
    python plot_results.py --method=$method \
                           --distribution=$distribution \
                           --correlation=$correlation \
                           --structure=$structure \
                           --mode=multi \
                           --parameters 5 0.05 \
                           --from_size=$from_size \
                           --to_size=$to_size \
                           --n_sample=$n_sample \
                           --n_restart=$n_restart
fi

#######################################################################################
## Dirichlet distribution
## CPC
#python structural_scores.py --method=cpc --distribution=dirichlet --structure=asia \
                            #--mode=multi --parameters 5 0.05 --from_size=$from_size \
                            #--to_size=$to_size --n_sample=$n_sample \
                            #--n_restart=$n_restart &&
#python plot_results.py --method=cpc --distribution=dirichlet --structure=asia  \
                       #--mode=multi --parameters 5 0.05 --from_size=$from_size \
                       #--to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart

## Elidan
#python structural_scores.py --method=elidan --distribution=dirichlet \
                            #--structure=asia --mode=multi --parameters 4 4 \
                            #--from_size=$from_size --to_size=$to_size \
                            #--n_sample=$n_sample --n_restart=$n_restart &&
#python plot_results.py --method=elidan --distribution=dirichlet --structure=asia \
                       #--mode=multi --parameters 4 4 --from_size=$from_size \
                       #--to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart
######################################################################################



######################################################################################
## Gaussian distribution
## CPC
#python structural_scores.py --method=cpc --distribution=gaussian --correlation=0.8 \
                            #--structure=asia --mode=multi --parameters 5 0.05 \
                            #--from_size=$from_size --to_size=$to_size \
                            #--n_sample=$n_sample --n_restart=$n_restart &&
#python plot_results.py --method=cpc --distribution=gaussian --correlation=0.8 \
                       #--structure=asia --mode=multi --parameters 5 0.05 \
                       #--from_size=$from_size --to_size=$to_size \
                       #--n_sample=$n_sample --n_restart=$n_restart

## Elidan
#python structural_scores.py --method=elidan --distribution=gaussian \
                            #--correlation=0.8 --structure=asia --mode=multi \
                            #--parameters 4 4 --from_size=$from_size \
                            #--to_size=$to_size --n_sample=$n_sample \
                            #--n_restart=$n_restart &&
#python plot_results.py --method=elidan --distribution=gaussian --correlation=0.8 \
                       #--structure=asia --mode=multi --parameters 4 4 \
                       #--from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart
######################################################################################



######################################################################################
## Student distribution
## CPC
#python structural_scores.py --method=cpc --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 5 0.05 \
                            #--from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
#python plot_results.py --method=cpc --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 5 0.05 \
                            #--from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart

## Elidan
#python structural_scores.py --method=elidan --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 4 4\
                            #--from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart &&
#python plot_results.py --method=elidan --distribution=student --correlation=0.8 --structure=asia --mode=multi --parameters 4 4 \
                            #--from_size=$from_size --to_size=$to_size --n_sample=$n_sample --n_restart=$n_restart
######################################################################################
