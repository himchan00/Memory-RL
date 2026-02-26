#!/bin/bash

# Define the ranges/lists for your parameters
MODELS=("mate" "mate_compile" "lstm" "lstm_compile" "gpt" "gpt_compile")
HIDDEN_SIZES=(128 256)
MAX_SEQS=(50 100 200 400 800 1600)
MODE=("rollout" "train")

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