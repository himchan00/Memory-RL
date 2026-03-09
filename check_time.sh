#!/bin/bash

# Define the ranges/lists for your parameters
MODELS=("mate" "lstm" "gpt")
HIDDEN_SIZES=(256)
MAX_SEQS=(100 200 300 400 500 600 700 800 900 1000)
MODE=("rollout" "update")

# Fixed parameters
N_LAYER=1
BATCH_SIZE=64

# Nested loops to iterate through all combinations
for MODEL in "${MODELS[@]}"; do
    for HIDDEN in "${HIDDEN_SIZES[@]}"; do
        for SEQ in "${MAX_SEQS[@]}"; do
            for M in "${MODE[@]}"; do
                echo "------------------------------------------------"
                echo "Running: Model=$MODEL, Hidden=$HIDDEN, MaxSeq=$SEQ, Mode=$M"
                echo "------------------------------------------------"

                # Execute the python script
                python check_time.py \
                    --model "$MODEL" \
                    --hidden_size "$HIDDEN" \
                    --n_layer "$N_LAYER" \
                    --max_seq_length "$SEQ" \
                    --batch_size "$BATCH_SIZE" \
                    --mode "$M"
            done
        done
    done
done