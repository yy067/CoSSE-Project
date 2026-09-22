# -*- coding : utf-8 -*-
# @FileName  : process_ham.py
# @Description: Preprocessing script for HAM10000 dataset (Patient-level Stratified Split)

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def process_ham10000(dataset_root_path):
    print("开始解析 HAM10000 临床数据与图像路径...")

    # 1. 读取元数据
    meta_path = os.path.join(dataset_root_path, "HAM10000_metadata.csv")
    df = pd.read_csv(meta_path)

    # 2. 扫描并构建所有图像的绝对路径映射 (搜索 part_1 和 part_2)
    all_imgs = glob.glob(os.path.join(dataset_root_path, "HAM10000_images_part_[12]", "*.jpg"))
    img_name_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in all_imgs}

    # 匹配图像路径到 dataframe
    df['img_path_full'] = df['image_id'].map(img_name_to_path)

    missing_count = df['img_path_full'].isna().sum()
    if missing_count > 0:
        print(f"[Warning] 有 {missing_count} 张图片在文件夹中未找到，将被移除。")
        df = df.dropna(subset=['img_path_full']).copy()

    # 3. 提取需要的临床列与标签列 (dx)
    features = ['dx_type', 'age', 'sex', 'localization']
    target = 'dx'
    cols_to_keep = ['lesion_id', 'image_id', 'img_path_full', target] + features
    data_df = df[cols_to_keep].copy()

    # 4. 缺失值处理
    # 年龄采用中位数填充，分类变量采用众数填充
    data_df['age'] = data_df['age'].fillna(data_df['age'].median())
    for col in ['dx_type', 'sex', 'localization']:
        data_df[col] = data_df[col].fillna(data_df[col].mode()[0])

    # 统一转换为字符串并大写，规范特征
    for col in ['dx_type', 'sex', 'localization']:
        data_df[col] = data_df[col].astype(str).str.strip().str.lower()

    # 5. 基于患者 ID (lesion_id) 提取【患者级】主标签，用于严谨的分层抽样
    patient_df = data_df.groupby('lesion_id')[target].agg(lambda x: x.mode()[0]).reset_index()
    X_patients = patient_df['lesion_id']
    y_patients = patient_df[target]

    print(f"总共有 {len(X_patients)} 位患者(病灶)，疾病分布为:\n{y_patients.value_counts().to_dict()}")

    # 6. 按照 60:20:20 的比例对患者进行分层划分 (先分出40%用于测试验证，再平分)
    train_patients, test_val_patients, _, y_test_val = train_test_split(
        X_patients, y_patients, test_size=0.40, random_state=42, stratify=y_patients
    )
    # 在等号左边增加 , _, _ 来接收被拆分出来的 y_val 和 y_test 标签
    val_patients, test_patients, _, _ = train_test_split(
        test_val_patients, y_test_val, test_size=0.50, random_state=42, stratify=y_test_val
    )

    # 7. 根据划分好的患者 ID 映射回完整的图像条目
    train_df = data_df[data_df['lesion_id'].isin(train_patients)].copy()
    val_df = data_df[data_df['lesion_id'].isin(val_patients)].copy()
    test_df = data_df[data_df['lesion_id'].isin(test_patients)].copy()

    # 8. 保存划分结果
    save_dir = os.path.join(dataset_root_path, "ProcessedMeta")
    os.makedirs(save_dir, exist_ok=True)

    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"处理完成！文件已成功保存至: {save_dir}")
    print(f"图像样本数 -> 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")


if __name__ == "__main__":
    ham_path = "/home/dlec/hyq/data/HAM10000"  # 你的实际路径
    process_ham10000(ham_path)