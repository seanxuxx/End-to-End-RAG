#!/bin/bash

embedding_models=("GIST-large-Embedding-v0" "bge-large-en-v1.5")
chunk_sizes=(500 1000 1500)
chunk_overlaps=(50 100 200)


for model in "${embedding_models[@]}"; do
    for size in "${chunk_sizes[@]}"; do
        for overlap in "${chunk_overlaps[@]}"; do
            echo "Running: experiment_pipeline.py with model=$model, chunk_size=$size, chunk_overlap=$overlap"
            python experiment_pipeline.py --embedding_model "$model" --chunk_size "$size" --chunk_overlap "$overlap"
        done
    done
done
