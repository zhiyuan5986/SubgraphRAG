dataset=webqsp
export SWANLAB_API_KEY=6xX60x1CzPEWbfdso1UwJ
python train_asl.py -d $dataset \
                --split train \
                --val_split val \
                --test_split "" \
                --data_dir . \
                --device cuda:0 \
                --seed 42 \
                --num_threads 16 \
                --topic_pe \
                --num_rounds 2 \
                --num_reverse_rounds 2 \
                --num_epochs 10000 \
                --patience 50 \
                --eval_k_list 50,100,200,400 \
                --gamma_pos 0.0 \
                --gamma_neg 2.0 \
                --neg_loss_weight 0.3 \
                --lr 1e-5 \
                --max_grad_norm 1.0 \
                --skip_no_path \
                --swanlab_mode cloud

