#!/bin/bash

# Define the ranges/lists for your parameters
MODELS=("mate" "lstm")
HIDDEN_SIZES=(128 256 512)
MAX_SEQS=(50 100 200 300 400 500 600 700 800 900 1000 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000)

# Fixed parameters
N_LAYER=1
BATCH_SIZE=256

# Nested loops to iterate through all combinations
for MODEL in "${MODELS[@]}"; do
    for HIDDEN in "${HIDDEN_SIZES[@]}"; do
        for SEQ in "${MAX_SEQS[@]}"; do
            
            echo "------------------------------------------------"
            echo "Running: Model=$MODEL, Hidden=$HIDDEN, MaxSeq=$SEQ"
            echo "------------------------------------------------"

            # Execute the python script
            python check_time.py \
                --model "$MODEL" \
                --hidden_size "$HIDDEN" \
                --n_layer "$N_LAYER" \
                --max_seq_length "$SEQ" \
                --batch_size "$BATCH_SIZE"
                
        done
    done
done