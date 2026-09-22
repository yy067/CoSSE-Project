import os
import random
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset_with_stratified_labels(image_dir, metadata_csv, gt_csv, output_dir,
                                         train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    # 加载元数据和标签数据
    metadata_df = pd.read_csv(metadata_csv)
    gt_df = pd.read_csv(gt_csv)

    # 将多标签转化为单一标签（每行只有一个1，取该列名作为类别）
    label_cols = gt_df.columns[1:]  # 去掉 image 列
    gt_df['label'] = gt_df[label_cols].idxmax(axis=1)  # 取最大值对应的列名作为类别

    # 确保 image 对应的文件名
    all_images = gt_df["image"].tolist()
    all_images_jpg = [img + ".jpg" for img in all_images]

    # 按标签分层划分
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images_jpg, gt_df['label'], stratify=gt_df['label'],
        test_size=(1 - train_ratio), random_state=seed
    )

    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)  # 从 temp 中划分 valid/test
    valid_imgs, test_imgs, valid_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, stratify=temp_labels,
        test_size=(1 - valid_ratio_adjusted), random_state=seed
    )

    # 建立映射
    image_to_split = {img[:-4]: 'train' for img in train_imgs}
    image_to_split.update({img[:-4]: 'valid' for img in valid_imgs})
    image_to_split.update({img[:-4]: 'test' for img in test_imgs})

    # 创建子目录
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # 复制图像
    for img_file in train_imgs + valid_imgs + test_imgs:
        src = os.path.join(image_dir, img_file)
        dst_dir = os.path.join(output_dir, image_to_split[img_file[:-4]])
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_dir, img_file))

    # 添加 split 信息
    metadata_df["split"] = metadata_df["image"].map(image_to_split)
    gt_df["split"] = gt_df["image"].map(image_to_split)

    # 保存
    for split in ["train", "valid", "test"]:
        metadata_df[metadata_df["split"] == split].to_csv(os.path.join(output_dir, f"{split}_metadata.csv"), index=False)
        gt_df[gt_df["split"] == split].to_csv(os.path.join(output_dir, f"{split}_labels.csv"), index=False)

    # 打印信息
    print(f"数据集分层抽样完成（seed={seed}）：")
    print(f"训练集：{len(train_imgs)} 张")
    print(f"验证集：{len(valid_imgs)} 张")
    print(f"测试集：{len(test_imgs)} 张")


if __name__ == "__main__":
    IMAGE_DIR = "./data/isic2019/ISIC_2019_Training_Input/"
    METADATA_CSV = "./data/isic2019/ISIC_2019_Training_Metadata.csv"
    GT_CSV = "./data/isic2019/ISIC_2019_Training_GroundTruth.csv"
    OUTPUT_DIR = "./data/isic2019/"

    split_dataset_with_stratified_labels(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        gt_csv=GT_CSV,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )