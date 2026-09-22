#  run isic
python main_classify63_ham.py --exp_name full_isic --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset isic2019 --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4 --w_imp 0.01 --use_instruct --max_epoch 30

#  run ham
python main_classify63_ham.py --exp_name full_ham --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset ham10000 --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4 --w_imp 0.01 --use_instruct --max_epoch 30

#  run food101
python main_classify63_ham.py --exp_name full_food101 --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset food --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4 --w_imp 0.01 --use_instruct

# Monitor the training process
python -m tensorboard.main --logdir ./logs --port 6006