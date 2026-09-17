
RANDOM_MODE="Randn"         # options: Randn / Randint / PGUXoR / PGUReuse

seeds=(15 23 37 42 56 69 73 84 92 100)

for seed in "${seeds[@]}"; do
    echo "Running experiment with seed: $seed"
    python snn_mlp.py \
        --random_mode $RANDOM_MODE \
        --seed $seed \
        --in_features 1024 \
        --hidden_features 128 \
        --out_features 10 \
        --time_step 16 \
        --batch_size 128 \
        --test_batch_size 1024 \
        --epochs 200 \
        --lr 2.5 \
        --eta_min 0.1 \
        --mu 1.0
    echo "Experiment with seed $seed completed"
done

