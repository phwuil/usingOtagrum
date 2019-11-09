#!/bin/bash

# One Script to rule them all, One Script to find them,
# One Script to bring them all and in the darkness bind them.

DATA_DIR_PREFIX="../data"
RES_DIR_PREFIX="../results"
FIG_DIR_PREFIX="../figures"

compute="all"
mode="multi"
method="cpc"
score="all"

distribution="gaussian"
correlation="0.8"
structure="asia"

sample_size=50000
test_size=1000

from_size=100
to_size=10000
n_sample=11
n_restart=1

from_n_node=2
to_n_node=22
node_step=5
density=1.2

mcss=5      # Maximum size for the conditioning set in continuous PC
alpha=0.05  # Confidence level for continuous PC

mp=4        # Maximum parent in Elidan learning
hcr=4       # Number of restart for the hill climbing in Elidan

sample_size_time=30000

parameters=

#data_exist=false
#results_exist=false
#figures_exist=false

forced=0
regenerate=0
recompute=0
replot=0


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
        -M | --method )
            shift
            method=$1;;
        --compute )
            shift
            compute=$1;;
        --score )
            shift
            score=$1;;
        --regenerate )
            regenerate=1;;
        --recompute )
            recompute=1;;
        --replot )
            replot=1;;
        -f )
            forced=1;;
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
TIME_DIR_SUFFIX="$distribution/time_complexity/r${correlation//./}"

if [ "$distribution" = "dirichlet" ]; then
    correlation=
    DIR_SUFFIX="$distribution/$structure"
fi

DATA_DIR="$DATA_DIR_PREFIX/samples/$DIR_SUFFIX"
STRUCT_DIR="$DATA_DIR_PREFIX/structures"
RES_DIR="$RES_DIR_PREFIX/$DIR_SUFFIX"
TIME_RES_DIR="$RES_DIR_PREFIX/$TIME_DIR_SUFFIX"
FIG_DIR="$FIG_DIR_PREFIX/$DIR_SUFFIX"
TIME_FIG_DIR="$FIG_DIR_PREFIX/$TIME_DIR_SUFFIX"

str_alpha=$(awk '{print 100*$1}' <<< "${alpha}")

TIME_FILE_NAME="time_${distribution}_f${from_n_node}t${to_n_node}s${node_step}"
TIME_FILE_NAME="${TIME_FILE_NAME}mcss${mcss}alpha${str_alpha}mp${mp}hcr${hcr}" 

FILE_NAME="${mode}_${method}_${structure}_${distribution}"
FILE_NAME="${FILE_NAME}_f${from_size}t${to_size}s${n_sample}r${n_restart}" 
if [ "$method" = "cpc" ]; then
    FILE_NAME="${FILE_NAME}mcss${mcss}alpha$(awk '{print 100*$1}' <<< "${alpha}")"
    parameters="$mcss $alpha"
elif [ "$method" = "elidan" ]; then
    FILE_NAME="${FILE_NAME}mp${mp}hcr${hcr}"
    parameters="$mp $hcr"
fi

# Checking if structure text file exists
if [ -f "$STRUCT_DIR/$structure.txt" ]; then
    echo "The file $structure.txt exists."
else
    echo "The file $structure.txt doesn't exist."
    exit 1
fi

if [ -d "$DATA_DIR" ]; then
    if [ $(ls -1 | wc -l) -lt $n_restart ]; then
        response=
        echo "There is less file than wanted number of restarts."
        echo "Generate more files? (y/n) > "
        read response
        if [ "$response" != "y" ]; then
            echo "Exiting program."
            exit 1
        fi
        python generate_data.py --distribution=$distribution \
                                --structure=$structure \
                                --n_sample=$n_restart \
                                --sample_size=$sample_size \
                                --correlation=$correlation
    fi
else
    echo "No corresponding data have been found. Generating some..."
    python generate_data.py --distribution=$distribution \
                            --structure=$structure \
                            --n_sample=$n_restart \
                            --sample_size=$sample_size \
                            --correlation=$correlation
fi

#####################################################################################
#                          Compute scores and loglikelihood results                 #
#####################################################################################

#if [ "$recompute" = "1" ]; then
    #if [ "$forced" != "1"]; then
        #response=
        #echo "Old data are going to be removed."
        #echo "Do you really want to proceed ? (y/n) > "
        #read response
        #if [ "$response" != "y" ]; then
            #echo "Exiting program."
            #exit 1
        #fi
    #fi
#fi

if [ "$compute" = "scores" ] || [ "$compute" = "all" ]; then
    if [ "$score" = "skeleton" ] || [ "$score" = "all" ]; then
        if [ -f "$RES_DIR/scores/${score}_scores_$FILE_NAME.csv" ] && [ "$recompute" = "0" ]; then
            echo "Result file for scores exists."
        else
            echo "Result file for scores does'nt exist."
            echo "Doing scientific stuff to generate one..."
            echo "Computing results for scores..."
            python structural_scores.py --method=$method \
                                        --distribution=$distribution \
                                        --correlation=$correlation \
                                        --structure=$structure \
                                        --score=$score \
                                        --mode=multi \
                                        --parameters $parameters \
                                        --from_size=$from_size \
                                        --to_size=$to_size \
                                        --n_sample=$n_sample \
                                        --n_restart=$n_restart
        fi
    fi
fi

if [ "$compute" = "loglikelihood" ] || [ "$compute" = "all" ]; then
    if [ -f "$RES_DIR/loglikelihood/loglikelihood_$FILE_NAME.csv" ] && [ "$recompute" = "0" ]; then
        echo "Result file for loglikelihood exists."
    else
        echo "Computing results for loglikelihood..."
        python loglikelihood_performances.py --method=$method \
                                             --distribution=$distribution \
                                             --correlation=$correlation \
                                             --structure=$structure \
                                             --mode=multi \
                                             --parameters $parameters \
                                             --from_size=$from_size \
                                             --to_size=$to_size \
                                             --n_sample=$n_sample \
                                             --n_restart=$n_restart \
                                             --test_size=$test_size
    fi
fi

if [ "$compute" = "time" ] || [ "$compute" = "all" ]; then
    if [ -f "$TIME_RES_DIR/$TIME_FILE_NAME.csv" ] && [ "$recompute" = "0" ]; then
        echo "Result file for time complexity exists."
    else
        echo "Computing results for time complexity..."
        python time_complexity.py --distribution=$distribution \
                                  --sample_size=$sample_size_time \
                                  --density=$density \
                                  --correlation=$correlation \
                                  --mcss=$mcss \
                                  --alpha=$alpha \
                                  --mp=$mp \
                                  --hcr=$hcr \
                                  --from_size=$from_n_node \
                                  --to_size=$to_n_node \
                                  --step=$node_step
    fi
fi


#####################################################################################
#                                  Plot figures                                     #
#####################################################################################

#if [ "$recompute" = "1" ]; then
    #if [ "$forced" != "1"]; then
        #response=
        #echo "Old data are going to be removed."
        #echo "Do you really want to proceed ? (y/n) > "
        #read response
        #if [ "$response" != "y" ]; then
            #echo "Exiting program."
            #exit 1
        #fi
    #fi
#fi

if [ "$compute" = "scores" ] || [ "$compute" = "all" ]; then
    if [ -f "$FIG_DIR/scores/${score}_scores_$FILE_NAME.pdf" ] && [ "$replot" = "0" ]; then
        echo "Figure file for scores exists."
    else
        echo "Plotting figure for scores..."
        python plot_scores.py --method=$method \
                               --score=$score \
                               --distribution=$distribution \
                               --correlation=$correlation \
                               --structure=$structure \
                               --mode=multi \
                               --mcss=$mcss \
                               --alpha=$alpha \
                               --mp=$mp \
                               --hcr=$hcr \
                               --from_size=$from_size \
                               --to_size=$to_size \
                               --n_sample=$n_sample \
                               --n_restart=$n_restart
    fi
fi

if [ "$compute" = "loglikelihood" ] || [ "$compute" = "all" ]; then
    if [ -f "$FIG_DIR/loglikelihood/loglikelihood_$FILE_NAME.pdf" ] && [ "$replot" = "0" ]; then
        echo "Figure file for loglikelihood exists."
    else
        echo "Plotting figure for loglikelihood..."
        python plot_loglikelihood.py --method=$method \
                                     --distribution=$distribution \
                                     --correlation=$correlation \
                                     --structure=$structure \
                                     --mode=multi \
                                     --parameters $parameters \
                                     --from_size=$from_size \
                                     --to_size=$to_size \
                                     --n_sample=$n_sample \
                                     --n_restart=$n_restart
    fi
fi

if [ "$compute" = "time" ] || [ "$compute" = "all" ]; then
    if [ -f "$TIME_FIG_DIR/$TIME_FILE_NAME.pdf" ] && [ "$replot" = "0" ]; then
        echo "Figure file for time exists."
    else
        echo "Plotting figure for time complexity..."
        python plot_time_complexity.py --distribution=$distribution \
                                       --sample_size=$sample_size_time \
                                       --density=$density \
                                       --correlation=$correlation \
                                       --mcss=$mcss \
                                       --alpha=$alpha \
                                       --mp=$mp \
                                       --hcr=$hcr \
                                       --from_size=$from_n_node \
                                       --to_size=$to_n_node \
                                       --step=$node_step
    fi
fi

