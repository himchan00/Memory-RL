#!/bin/bash

# Define the ranges/lists for your parameters
MODELS=("mate" "lstm" "gpt" "splagger" "mamba")
HIDDEN_SIZES=(128)
MAX_SEQS=(100 200 300 400 500 600 700 800 900 1000)
MODE=("rollout" "update")

# Fixed parameters
N_LAYER=1
BATCH_SIZE=64
EMBEDDER_TYPE=gpt_ffn  # mate only

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
                    --embedder_type "$EMBEDDER_TYPE" \
                    --mode "$M"
            done
        done
    done
done