#!/bin/bash

embedding_models=("all-mpnet-base-v2")
chunk_sizes=(500 1000 1500)
chunk_overlaps=(50 100 200)


for model in "${embedding_models[@]}"; do
    for size in "${chunk_sizes[@]}"; do
        for overlap in "${chunk_overlaps[@]}"; do
            # Skip the combination (chunk_size=1000, chunk_overlap=100)
            if [[ "$size" -eq 1000 && "$overlap" -eq 100 ]]; then
                echo "Skipping: chunk_size=1000, chunk_overlap=100"
                continue
            fi
            
            echo "Running: experiment_pipeline.py with model=$model, chunk_size=$size, chunk_overlap=$overlap"
            python experiment_pipeline.py --embedding_model "$model" --chunk_size "$size" --chunk_overlap "$overlap"
        done
    done
done
