import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_args():
    parser = argparse.ArgumentParser(description="Generate ISIC2019 descriptions using various LLMs")
    parser.add_argument(
        "--model",
        type=str,
        default="phi3",
        choices=["phi3", "phi4", "llama3", "qwen"],
        help="Which LLM to use for generation."
    )
    return parser.parse_args()


def main():
    args = get_args()

    MODEL_MAP = {
        "phi3": "microsoft/Phi-3-small-8k-instruct",
        "phi4": "microsoft/Phi-4-mini-instruct",
        "llama3": "NousResearch/Meta-Llama-3-8B-Instruct",
        "qwen": "Qwen/Qwen2.5-7B-Instruct"
    }

    model_id = MODEL_MAP[args.model]

    print(f"正在加载模型 [{model_id}] 和分词器，请稍候...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    input_file = "./data/isic2019/labels.txt"
    output_file = f"./data/isic2019/class_descriptions_{args.model}.json"

    with open(input_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    print(f"成功读取 {len(labels)} 个类别标签。")

    descriptions = {}

    isic_mapping = {
        "MEL": "Melanoma",
        "NV": "Melanocytic nevus",
        "BCC": "Basal cell carcinoma",
        "AK": "Actinic keratosis",
        "BKL": "Benign keratosis",
        "DF": "Dermatofibroma",
        "VASC": "Vascular lesion",
        "SCC": "Squamous cell carcinoma",
        "UNK": "Unknown skin lesion"
    }

    for label in labels:
        clean_label = isic_mapping.get(label.upper(), label.replace("_", " "))

        system_prompt = (
            f"You are an expert Dermatologist and Medical Computer Vision Researcher. "
            f"In one or two sentences, describe the typical dermoscopic visual features of '{clean_label}'. "
            f"Focus STRICTLY on visible morphological and dermatoscopic patterns ONLY: shape asymmetry, border irregularity, "
            f"color variegation (e.g., blue-white veil, dark brown), and specific structural patterns (e.g., pigment networks, dots, globules, streaks, blood vessels). "
            f"DO NOT mention patient symptoms, clinical history, etiology, anatomical location, or treatments. "
            f"Output the description directly, keep it highly objective and precise."
        )

        messages = [
            {"role": "user", "content": system_prompt}
        ]

        outputs = pipe(
            messages,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=True,
        )

        try:
            generated_text = outputs[0]['generated_text'][-1]['content'].strip()
        except (KeyError, TypeError):
            generated_text = outputs[0]['generated_text'].replace(system_prompt, "").strip()

        descriptions[label] = generated_text
        print(f"[{label}] 完成 -> {generated_text}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=4)

    print(f"\n[{args.model}] 所有描述已成功保存至 {output_file}！")


if __name__ == "__main__":
    main()