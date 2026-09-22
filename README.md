# CoSSE-Project

Implementation of CoSSE for image-text multimodal classification.

## Data Processing

Preprocess each dataset using the corresponding script before training.

### ISIC2019

```bash
python process_isic.py
```

### HAM10000

```bash
python process_ham.py
```

### UPMC-Food-101

```bash
python process_food_101.py
```

## Training

### ISIC2019

To train CoSSE on the ISIC2019 dataset, run:

```bash
python main_classify.py --exp_name full_isic --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset isic2019 --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4 --w_imp 0.01 --use_instruct --max_epoch 30

```

### HAM10000

To train CoSSE on the HAM10000 dataset, run:

```bash
python main_classify.py --exp_name full_ham --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset ham10000 --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4 --w_imp 0.01 --use_instruct --max_epoch 30

```

### UPMC-Food-101

To train CoSSE on the UPMC-Food-101 dataset, run:

```bash
python main_classify.py --exp_name full_food101 --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset food --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4 --w_imp 0.01 --use_instruct

```
