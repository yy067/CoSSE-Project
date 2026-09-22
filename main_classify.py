# general libs
import os,  argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Dict
import warnings


warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")
from utils import *
from models import swin, bert, classifier

from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl

from utils.datasets import create_loaders

from sklearn.metrics import f1_score, precision_score, recall_score

import pandas as pd

from pytorch_lightning.callbacks import EarlyStopping

class MetaEmbeddings(nn.Module):
    def __init__(self, site_dim=9, sex_dim=2, output_dim=768, dropout=0.1):
        super().__init__()
        self.age_embedding = nn.Linear(1, output_dim)
        self.sex_embedding = nn.Linear(1, output_dim)
        self.site_embedding = nn.Linear(1, output_dim)

        self.pe_age = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.pe_sex = nn.Parameter(torch.zeros(1, sex_dim, output_dim))
        self.pe_site = nn.Parameter(torch.zeros(1, site_dim, output_dim))

        self.dropout_age = nn.Dropout(dropout)
        self.dropout_sex = nn.Dropout(dropout)
        self.dropout_site = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 1 + sex_dim + site_dim]
        sex_dim = self.pe_sex.shape[1]

        age = x[:, 0:1]  # (B, 1)
        sex = x[:, 1:1 + sex_dim]  # (B, sex_dim)
        site = x[:, 1 + sex_dim:]  # (B, site_dim)

        age_emb = self.dropout_age(self.age_embedding(age.unsqueeze(-1)) + self.pe_age)
        sex_emb = self.dropout_sex(self.sex_embedding(sex.unsqueeze(-1)) + self.pe_sex)
        site_emb = self.dropout_site(self.site_embedding(site.unsqueeze(-1)) + self.pe_site)

        meta_emb = torch.cat([age_emb, sex_emb, site_emb], dim=1)

        meta_feat = meta_emb.mean(dim=1)

        return meta_feat


class MetaClassifier(nn.Module):
    def __init__(self, output_dim=768, site_dim=9, sex_dim=2):
        super().__init__()
        self.backbone = MetaEmbeddings(site_dim=site_dim, sex_dim=sex_dim, output_dim=output_dim)

    def forward(self, x, dump_attn=False, return_features=False, **kwargs):
        feat = self.backbone(x)

        if return_features:
            return None, feat

        return feat

    def freeze_backbone(self):
        pass

    def requires_grad_(self, requires_grad=True):
        return super().requires_grad_(True)

class HamMetaEmbeddings(nn.Module):
    def __init__(self, dims, output_dim=768, dropout=0.1):
        super().__init__()
        self.dims = dims
        self.total_tokens = sum(dims)

        self.embeddings = nn.ModuleList([nn.Linear(1, output_dim) for _ in dims])
        self.pes = nn.ParameterList([nn.Parameter(torch.zeros(1, d, output_dim)) for d in dims])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in dims])

    def forward(self, x):
        feat_list = []
        idx = 0
        for i, d in enumerate(self.dims):
            feat_slice = x[:, idx:idx + d]
            idx += d

            emb = self.dropouts[i](self.embeddings[i](feat_slice.unsqueeze(-1)) + self.pes[i])
            feat_list.append(emb)


        meta_emb = torch.cat(feat_list, dim=1)

        meta_feat = meta_emb.mean(dim=1)
        return meta_feat

class HamMetaClassifier(nn.Module):
    def __init__(self, output_dim=768, dims=[]):
        super().__init__()
        self.backbone = HamMetaEmbeddings(dims=dims, output_dim=output_dim)

    def forward(self, x, dump_attn=False, return_features=False, **kwargs):
        feat = self.backbone(x)
        if return_features:
            return None, feat
        return feat

    def freeze_backbone(self):
        pass

    def requires_grad_(self, requires_grad=True):
        return super().requires_grad_(True)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="food101",
        choices=["food", "isic2019", "ham10000"],
        help="which dataset to use.",
    )



    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Set the maximum batch size for training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of workers for pytorch's dataloader.",
    )


    # Encoder

    # General
    parser.add_argument("--name", default="", type=str, help="model name")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )

    parser.add_argument(
        "--max_epoch",
        type=int,
        # nargs="+",
        default=10,
        help="max num of epoches for training",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "--exp_name",
        default="model",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: model)",
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )



    # Optimisers
    parser.add_argument(
        "--lr_vis", type=float, default=4e-4, help="Learning rate for visual encoder."
    )
    parser.add_argument(
        "--lr_text", type=float, default=5e-4, help="Learning rate for text encoder."
    )

    parser.add_argument(
        "--wd_vis", type=float, default=1e-3, help="Weight decay for visual encoder."
    )
    parser.add_argument(
        "--wd_text", type=float, default=1e-3, help="Weight decay for text encoder."
    )

    # parser.add_argument("--lamda", type=float, default=LAMDA, help="Lamda for L1 norm.")
    parser.add_argument(
        "--warmup_epochs", type=float, default=1, help="warmup epochs for lr scheduler"
    )
    # parser.add_argument('-t', '--bn-threshold', type=float, default=BN_threshold,
    #                     help='Threshold for slimming BNs.')
    parser.add_argument(
        "--backbone",
        default="swinb_224",
        type=str,
        choices=["swinb_224", "swinb_384", "vitb"],
    )

    # ---------------model setting----------------
    parser.add_argument(
        "--fuse_method",
        default="late_concat",
        type=str,
        choices=[
            "late_concat",
            "instruct_v2t",
            "instruct_t2v",
            "instruct_moe_t2v",
            "instruct_moe_v2t",
            "instruct_mm_moe_t2v",
            "mope",
            "sequentialfuse",
            "img_only",
            "text_only",
            "p_sequential",
        ],
        help="how to fuse to modality",
    )
    parser.add_argument(
        "--train_instructor",
        action="store_true",
        default=False,
        help="whether the instructor should be trained at the same time.",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Whether to freeze (vision) encoder.",
    )
    parser.add_argument(   
        "--route_per_layer",
        action="store_true",
        default=True,
        help="whether to learn a routing weight for each layers",
    )
    parser.add_argument(
        "--dense_routing",
        action="store_true",
        default=True,
        help="whether to densely route expert (all experts are used)). temporarily deprecated arg",
    )
    # ----------------prompt learning---------------
    parser.add_argument(
        "--use_vpt", action="store_true", default=False, help="Whether to use VPT."
    )
    parser.add_argument(
        "--use_pbert",
        action="store_true",
        default=False,
        help="Whether to use Prompted Bert.",
    )

    parser.add_argument(
        "--vis_prompt_type",
        default="vpt",
        choices=["vpt"],
        type=str,
        help="how to apply prompt tuning?",
    )

    parser.add_argument(
        "--d_cross", type=int, default=8, help="dimension of cross-feature embd"
    )
    parser.add_argument(
        "--d_inter", type=int, default=2, help="dimension of cross-feature embd"
    )


    parser.add_argument(
        "--moe_n_experts", type=int, default=4, help="number of experts for moe"
    )
   
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=6,
        help="number of learnable visual prompts",
    )
    parser.add_argument(
        "--t_prompt_length",
        type=int,
        default=4,
        help="number of learnable text prompts",
    )

    parser.add_argument(
        "--use_static_prompt",
        action="store_true",
        default=True,
        help="whether to use additional static visual prompt, temporarily deprecated arg (always true)",
    )
    parser.add_argument(
        "--use_instruct",
        action="store_true",
        default=True,
        help="whether to use instructor, temporarily deprecated arg (always true)",
    )
    parser.add_argument(
        "--moe_top_k", type=int, default=1, help="number of top k experts to use, has to be used together with dense_routing, temporarily deprecated arg"
    )
    parser.add_argument(
        "--prompt_init",
        type=str,
        default="uniform",
        choices=["uniform", "normal", "othorgonal"],
        help="how prompt and experts are inited",
    )
    # --------------loss---------------
    
    parser.add_argument(
        "--w_imp", type=float, default=0.01, help="weight for importance loss"
    )
    parser.add_argument(
        "--smooth_label",
        action="store_true",
        default=False,
        help="whether to use lable smoothing",
    )
    parser.add_argument(
        "--w_othor", type=float, default=0.0, help="weight for othor loss"
    )
    parser.add_argument(
        "--w_contrast", type=float, default=0.0, help="weight for contrastive loss"
    )

    # ---------------debug------------------
    parser.add_argument("--exp_note", default="", type=str, help="experiment note")
    # ----------experiment setting----------------
    parser.add_argument(
        "--n_shot", default=0, type=int, help="number of shots for low shot learning"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="whether to finetune the model",
    )

    parser.add_argument('--ignore_label', type=int, default=-1, help='ignore index for loss computation (default: -1)')

    return parser.parse_args()

class Model(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.validation_step_outputs = []
        # if args is a dictionary, convert to Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)

        self.save_hyperparameters(self.args)
        if self.args.dataset == "food":
            self.num_classes = 101

        elif self.args.dataset == "isic2019":
            self.num_classes = 8
        elif self.args.dataset == "ham10000":
            self.num_classes = 7
        else:
            self.num_classes = 3

        if self.args.dataset == "food":
            self.w_align = 0.4 # 0.4
        else:
            self.w_align = 0.5

        if self.args.dataset == "food":
            self.final_act = lambda x: F.log_softmax(x, dim=1)
            self.loss = nn.NLLLoss(ignore_index=self.args.ignore_label)

        elif self.args.dataset == "isic2019":
            self.final_act = lambda x: F.log_softmax(x, dim=1)

            train_csv_path = "./data/isic2019/train_labels.csv"

            if os.path.exists(train_csv_path):

                df = pd.read_csv(train_csv_path, sep=None, engine='python')

                classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

                class_counts = df[classes].sum().values
                total_samples = class_counts.sum()
                num_classes = len(classes)

                max_count = class_counts.max()
                weights = max_count / class_counts

                class_weights = torch.tensor(weights, dtype=torch.float32)

                print(f"[*] Successfully calculated ISIC class weights from {train_csv_path}")
                print(f"    Sample distribution: {dict(zip(classes, class_counts))}")
                print(f"    Calculated weights: {dict(zip(classes, np.round(weights, 4)))}")
            else:
                print(f"[Warning] {train_csv_path} not found.")
                class_weights = None

            self.loss = nn.NLLLoss(weight=class_weights, ignore_index=self.args.ignore_label)
        elif self.args.dataset == "ham10000":
            self.final_act = lambda x: F.log_softmax(x, dim=1)
            train_csv_path = "./data/HAM10000/ProcessedMeta/train.csv"

            if os.path.exists(train_csv_path):
                df = pd.read_csv(train_csv_path)
                class_counts = df['dx'].value_counts().sort_index().values
                max_count = class_counts.max()
                weights = max_count / class_counts
                class_weights = torch.tensor(weights, dtype=torch.float32)

                features = ['dx_type', 'sex', 'localization']

                self.ham_dims = [1] + [len(pd.get_dummies(df[col]).columns) for col in features]


                print(f"[*] Successfully initialized HAM10000 class and clinical weights from {train_csv_path}")
                print(f"    True sample distribution: {df['dx'].value_counts().to_dict()}")
                print(f"    Tabular input one-hot feature dimensions: {self.ham_dims}")
            else:
                print(f"[Warning] 找不到 {train_csv_path}，加载默认配置！")
                class_weights = None
                self.ham_dims = [1, 4, 3, 15]

            self.loss = nn.NLLLoss(weight=class_weights, ignore_index=self.args.ignore_label)

        self.vision_classifier = swin.get_swin_classifier(
            num_classes=self.num_classes,
            backbone=self.args.backbone,
            use_vpt=self.args.use_vpt,
            moe_n_experts=self.args.moe_n_experts,
            prompt_length=self.args.prompt_length,
            use_static_prompt=self.args.use_static_prompt,
            prompt_init=self.args.prompt_init,
            use_instruct=self.args.use_instruct,
            d_cross=self.args.d_cross,
            d_inter=self.args.d_inter,
        )

        text_hidden_dim = 768
        if self.args.dataset == "isic2019":

            self.text_classifier = MetaClassifier(
                output_dim=text_hidden_dim,
                site_dim=9,
                sex_dim=2
            )

        elif self.args.dataset == "ham10000":
            self.text_classifier = HamMetaClassifier(
                output_dim=text_hidden_dim,
                dims=self.ham_dims
            )
        else:
            self.text_classifier = bert.BertClassifier(
                self.num_classes,
                use_prompt=self.args.use_pbert,
                prompt_length=self.args.t_prompt_length,
            )

        self.fuse_method = self.args.fuse_method
        self.classifier = classifier.VisionTextClassifiers(
            self.vision_classifier,
            self.text_classifier,
            self.num_classes,
            fusion_method=self.fuse_method,
            train_instructor=self.args.train_instructor,
            moe_n_experts=self.args.moe_n_experts,
            dense_routing=self.args.dense_routing,
            moe_top_k=self.args.moe_top_k,
            route_per_layer=self.args.route_per_layer,
        )
        if not self.args.train_instructor:
            self.text_classifier.requires_grad_(False)
        if self.args.finetune:
            self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(True)
        if self.fuse_method == "promptfuse":
            self.text_classifier.requires_grad_(True)
            self.text_classifier.freeze_backbone()
            self.vision_classifier.requires_grad_(False)
        if self.args.fuse_method == "sequentialfuse":
            self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(True)
        if self.args.fuse_method == "img_only":
            # self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(False)
        if self.args.fuse_method == "text_only":
            self.vision_classifier.requires_grad_(False)
            # self.text_classifier.requires_grad_(True)
        if self.args.fuse_method == "p_sequential":
            self.text_classifier.requires_grad_(True)
            self.text_classifier.freeze_backbone()
        if self.fuse_method == "instruct_v2t":
            self.text_classifier.freeze_backbone()
        # self.text_classifier.freeze_backbone()
        # self.vision_classifier.freeze_backbone()
        # TODO Sep 12: deprecated when prompt tuning, always freeze encoder
        if self.args.use_vpt and not self.args.freeze_encoder:
            print("[Warning] Using VPT without freezing encoder")

        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                num_params = np.prod(param.size())
                num_params_k = num_params / 1000
                print("{}, num_params: {}K".format(name, num_params_k))


        embed_dim = 128
        self.align_projs = nn.ModuleList([
            nn.Linear(embed_dim, 768),
            nn.Linear(embed_dim * 2, 768),
            nn.Linear(embed_dim * 4, 768),
            nn.Linear(embed_dim * 8, 768)
        ])


        hard_prompts = self._prepare_hard_prompts(self.args.dataset)
        if hard_prompts is not None:

            self.register_buffer("hard_prompt_features", hard_prompts)
        else:
            self.hard_prompt_features = None

    def _prepare_hard_prompts(self, dataset):
        import json
        import os

        if dataset == "food":
            json_path = "./data/food101/class_descriptions.json"
        elif dataset == "isic2019":
            json_path = "./data/isic2019/class_descriptions.json"
        elif dataset == "ham10000":
            json_path = "./data/HAM10000/class_descriptions.json"
        else:
            return None

        if not os.path.exists(json_path):
            print(f"[Warning] LLM hard prompts file not found at {json_path}. Alignment loss will not be used.")
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            descriptions_dict = json.load(f)

        descriptions_list = list(descriptions_dict.values())

        print(f"[*] Loaded {len(descriptions_list)} hard prompts. Extracting features using BertClassifier...")

        if dataset in ["isic2019", "ham10000"]:
            print("Instantiating temporary BertClassifier for hard prompts...")
            encoder_to_use = bert.BertClassifier(self.num_classes, use_prompt=False, prompt_length=0).cuda()
        else:
            encoder_to_use = self.text_classifier
        encoder_to_use.eval()

        with torch.no_grad():
            batch_size = 16
            all_cls_features = []

            for i in range(0, len(descriptions_list), batch_size):
                batch_texts = descriptions_list[i: i + batch_size]

                _, batch_cls = encoder_to_use(
                    text_input=batch_texts,
                    return_features=True
                )
                all_cls_features.append(batch_cls)

            hard_features_tensor = torch.cat(all_cls_features, dim=0)

        print(f"Successfully extracted hard prompt features, shape: {hard_features_tensor.shape}")

        return hard_features_tensor.detach()

    def training_step(self, batch, batch_idx):
        text_input, img_input, gt_label = batch
        # Compute outputs
        outputs, extra_out = self.classifier(img_input, text_input)

        if self.args.dataset in ["food", "snli", "isic2019", "ham10000"]:
            outputs = self.final_act(outputs)
        loss_val = self.loss(outputs, gt_label.squeeze())
        if extra_out is not None:
            if "importance_loss" in extra_out:
                imp_loss = extra_out["importance_loss"]
                self.log("imp_loss", imp_loss)
                loss_val += imp_loss * self.args.w_imp
            if "othor_loss" in extra_out:
                othor_loss = extra_out["othor_loss"]
                self.log("othor_loss", othor_loss)
                loss_val += othor_loss * self.args.w_othor

            if "dynamic_prompts" in extra_out and self.hard_prompt_features is not None:
                align_loss = 0.0

                target_hard_prompts = self.hard_prompt_features[gt_label.squeeze().long()]

                layer_weights = [1.0, 1.0, 1.0, 1.0]

                for i, layer_dynamic_prompt in enumerate(extra_out["dynamic_prompts"]):

                    if layer_dynamic_prompt is None or layer_weights[i] == 0.0:
                        continue

                    pooled_soft_prompt = layer_dynamic_prompt.mean(dim=1)

                    projected_soft_prompt = self.align_projs[i](pooled_soft_prompt)

                    cos_sim = F.cosine_similarity(projected_soft_prompt, target_hard_prompts, dim=-1)
                    layer_loss = (1.0 - cos_sim).mean()

                    align_loss += layer_loss * layer_weights[i]

                align_loss = align_loss / len(extra_out["dynamic_prompts"])
                self.log("align_loss", align_loss)

                loss_val += align_loss * self.w_align

        crt_vision_lr = self.optimizers().param_groups[0]["lr"]
        self.log("vision_lr", crt_vision_lr)
        self.log("train_loss", loss_val)

        # train batch stats
        with torch.no_grad():

            if self.args.dataset in ["food", "isic2019", "ham10000"]:
                outputs = self.final_act(outputs)  # for metric calc
                pred_label = torch.argmax(outputs, dim=1)
                correct_cnt = torch.sum(pred_label == gt_label.squeeze()).item()
                all_cnt = pred_label.size(0)
                acc = correct_cnt / all_cnt
                self.log("train_acc", acc)

            return loss_val

    def validation_step(self, batch, batch_idx):
        text_input, img_input, gt_label = batch
        # Compute outputs
        outputs, extra_out = self.classifier(img_input, text_input)
        outputs = self.final_act(outputs)
        # loss_val = self.loss(outputs, label.squeeze())
        if self.args.dataset in ["food", "isic2019", "ham10000"]:
            pred_label = torch.argmax(outputs, dim=1)

        ret_dict = {
            "pred_label": pred_label,
            "gt_label": gt_label.squeeze(),
            "text_input": text_input,
            "img_input": img_input,
        }

        if extra_out is not None:
            if "moe_scores" in extra_out:
                ret_dict["moe_scores"] = extra_out["moe_scores"]
            if "cls_" in extra_out:
                ret_dict["cls"] = extra_out["cls_"].detach().cpu()

        self.validation_step_outputs.append(ret_dict)
        return ret_dict

    def on_validation_epoch_end(self) -> None:

        if self.args.dataset in ["food", "isic2019", "ham10000"]:
            all_cnt = 0
            correct_cnt = 0
            expert_img_dir = os.path.join("debug", "route")
            os.makedirs(expert_img_dir, exist_ok=True)
            cls_features = []
            gt_labels = []
            all_preds = []

            for step_out in self.validation_step_outputs:
                pred_label = step_out["pred_label"]
                gt_label = step_out["gt_label"]
                # loss_val = step_out["loss"]
                all_cnt += pred_label.size(0)
                correct_cnt += torch.sum(pred_label == gt_label).item()

                if "cls" in step_out:
                    cls_features.append(step_out["cls"])
                gt_labels.append(gt_label)
                all_preds.append(pred_label)
            acc = correct_cnt / all_cnt
            self.log("val_acc", acc)

        if self.args.dataset in ["isic2019", "ham10000", "food"]:

            gts_np = torch.cat(gt_labels).cpu().numpy()
            preds_np = torch.cat(all_preds).cpu().numpy()

            precision = precision_score(gts_np, preds_np, average="macro", zero_division=0)
            recall = recall_score(gts_np, preds_np, average="macro", zero_division=0)
            f1_macro = f1_score(gts_np, preds_np, average="macro", zero_division=0)

            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1_macro", f1_macro)

        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Any:

        vision_lr = self.args.lr_vis
        text_lr = self.args.lr_text

        optimizer_cfg = [
            {
                "params": self.vision_classifier.parameters(),
                "lr": vision_lr,
                "weight_decay": self.args.wd_vis,
            },
            {
                "params": self.text_classifier.parameters(),
                "lr": text_lr,
                "weight_decay": self.args.wd_text,
            },
            {
                "params": self.align_projs.parameters(),
                "lr": vision_lr,
                "weight_decay": self.args.wd_vis,
            }
        ]
        # todo refactor as returned dict of params from init of the classifier
        if self.fuse_method == "late_concat":
            optimizer_cfg.append(
                {
                    "params": self.classifier.fusion_head.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_t2v":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_v2t":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": text_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_moe_t2v":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
            optimizer_cfg.append(
                {
                    "params": self.classifier.moe_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_moe_v2t":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
            optimizer_cfg.append(
                {
                    "params": self.classifier.moe_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "promptfuse":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "p_sequential":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": text_lr,
                    "weight_decay": 1e-4,
                }
            )
        optimizer = torch.optim.AdamW(optimizer_cfg)
        # optimizer = ScheduledOptim(optimizer,)

        if self.args.dataset == "food":
            milestones = [3, 6]
        elif self.args.dataset == "isic2019":
            milestones = [12, 22]
        elif self.args.dataset == "ham10000":
            milestones = [15, 24]
        else:
            milestones = [3, 6]
        # step lr,
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.4
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from model ckpt."""
        self.args = args

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove the frozen parameter
        filter = ["attention", "mlp", "attn", "downsample", "intermediate"]
        for k in list(checkpoint["state_dict"].keys()):
            for f in filter:
                if f in k:
                    del checkpoint["state_dict"][k]

        return super().on_save_checkpoint(checkpoint)

    def benchmark_memory(self):
        # do forward pass with batch size 1 for 10000 times
        # measure the memory
        # ensure cuda classifer
        self.classifier.to("cuda").eval()
        dummy_input_img = torch.randn(16, 3, 224, 224).cuda()

        if self.args.dataset == "isic2019":
            dummy_input_text = torch.randn(16, 12).cuda()
        elif self.args.dataset == "ham10000":
            total_dims = sum(self.ham_dims)
            dummy_input_text = torch.randn(16, total_dims).cuda()
        else:
            dummy_input_text = ["test"] * 16
        peak_mems = []
        self.eval()
        # optim = self.configure_optimizers()["optimizer"]
        device = torch.cuda.current_device()
        # warm up
        for i in range(2):
            out, _ = self.classifier(dummy_input_img, dummy_input_text)
            dummy_loss = out.sum()
            # dummy_loss.backward()
            # optim.step()
        for i in range(50):
            # optim.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

            # Perform inference
            with torch.no_grad():
                out, __ = self.classifier(dummy_input_img, dummy_input_text)
            # dummy_loss = out.sum()
            # dummy_loss.backward()
            # optim.step()
            # Measure memory after inference
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mems.append(peak_mem)
            torch.cuda.empty_cache()
            # measure the memory
        # print stat
        print(f"Mean peak memory: {np.mean(peak_mems)} MB")
        print(f"Std peak memory: {np.std(peak_mems)} MB")

    def benchmark_inference_speed(self):
        # do forward pass with batch size 1 for 10000 times
        # measure the time
        dummy_input_img = torch.randn(16, 3, 224, 224).cuda()

        if self.args.dataset == "isic2019":
            dummy_input_text = torch.randn(16, 12).cuda()
        elif self.args.dataset == "ham10000":

            total_dims = sum(self.ham_dims)
            dummy_input_text = torch.randn(16, total_dims).cuda()
        else:
            dummy_input_text = ["test"] * 16
        self.eval()
        self.classifier.eval()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        self.classifier.to("cuda")
        repetitions = 1000
        with torch.no_grad():  # warm up
            for i in range(50):
                self.classifier(dummy_input_img, dummy_input_text)
        print("Start timing...")
        timings = []
        with torch.no_grad():
            torch.cuda.synchronize()
            for i in range(repetitions):
                starter.record()
                self.classifier(dummy_input_img, dummy_input_text)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
            torch.cuda.synchronize()
        print(f"Mean time: {np.mean(timings)} ms")
        print(f"Std time: {np.std(timings)} ms")


def main():
    args = get_arguments()
    seed_everything(args.random_seed)

    if args.dataset == "food":
        data_path = "data/food101"
    elif args.dataset == "isic2019":
        data_path = "data/isic2019"
    elif args.dataset == "ham10000":
        data_path = "data/HAM10000"
    if args.dataset == "food":
        batch_size = 38
        num_workers = 12
    elif args.dataset == "isic2019":
        batch_size = 16
        num_workers = 12
    elif args.dataset == "ham10000":
        batch_size = 16
        num_workers = 12
    # batch_size = 64 if args.dataset == "snli" else 32
    batch_size_arg = args.batch_size
    batch_size = min(batch_size, batch_size_arg)

    train_loader, val_loader, test_loader = create_loaders(
        data_path,
        batch_size,
        num_workers,  # 修改
        args.n_shot,
    )

    model = Model(args)
    if args.dataset == "food":
        save_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            filename="{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            mode="max",
        )
    elif args.dataset in ["isic2019", "ham10000"]:
        save_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_f1_macro",
            filename="{epoch:02d}-{val_f1_macro:.4f}",
            save_top_k=1,
            mode="max",
        )
    logger = pl.loggers.TensorBoardLogger("logs", name=args.exp_name)

    max_epoch = args.max_epoch

    trainer = Trainer(
        accelerator="gpu",
        callbacks=[save_callback],
        precision=16,
        logger=logger,
        max_epochs=max_epoch,
        val_check_interval=0.33,
        # check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=1.0,
    )



    if args.evaluate:
        if args.ckpt is not None and args.ckpt != "":

            model = Model.load_from_checkpoint(args.ckpt, strict=False)
            model.overwrite_args(args)
            # save all scripts to logger dir
            model.eval()
            trainer.validate(model, test_loader)
        else:
            raise Warning("Trying to evaluate model but with no checkpoint provided") 
        model.benchmark_inference_speed()
        model.benchmark_memory()
    else:

        if args.ckpt is not None and args.ckpt != "":
            model = Model.load_from_checkpoint(args.ckpt, strict=False)


        os.system("cp -r *py models utils %s" % logger.log_dir)


        print("Training phase started...")
        trainer.fit(model, train_loader, val_loader)
        print("Training phase completed.")


        final_ckpt_path = os.path.join(logger.log_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_ckpt_path)
        print(f"Final backup model saved to {final_ckpt_path}")

        print("Automatic testing with the BEST validation checkpoint...")

        seed_everything(args.random_seed)

        best_ckpt_path = save_callback.best_model_path
        print(f"Found best checkpoint at: {best_ckpt_path}")

        if best_ckpt_path:
            best_model = Model.load_from_checkpoint(best_ckpt_path, strict=False)
            trainer.validate(best_model, dataloaders=test_loader)
        else:
            print("Warning: No best checkpoint found, validating with current model.")
            trainer.validate(model, dataloaders=test_loader)
        print("Full pipeline finished!")

if __name__ == "__main__":

    main()
