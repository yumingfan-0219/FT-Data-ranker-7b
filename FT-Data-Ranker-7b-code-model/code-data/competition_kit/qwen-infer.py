import os
import json
import torch
import platform
import openpyxl
from openpyxl import Workbook
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
#123
def contains_chinese(str):
    for char in str:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def process_file(file_path, filename, model, tokenizer, num_entries=20000):
    input_data = []
    output_data = []
    device = next(model.parameters()).device
    filename=file_path+filename
    with open(filename, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for item in tqdm(content[:num_entries]):
            input_text = item.get('instruction') + item.get('input')
            if contains_chinese(input_text):#不使用中文数据
                continue
            messages = input_text
            response, _ = model.chat(tokenizer, messages, history=[])
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            input_data.append(input_text)
            output_data.append(response)

    return input_data, output_data


def init_model():
    model_path = "Qwen-14B-0925/Qwen-14B-Chat"#更换为qwen14b路径
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        top_p=0,
        temperature=0,
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    file_path = 'FT-Data-Ranker-7b-code-model/code-data/data'#更换为实际存储路径
    files = [
        'train_bloom-c4-temp.json'
        ]

    for file in files:
        print(f"Processing {file} ...")
        input_data, output_data = process_file(file_path, file, model, tokenizer)
        json_file_path = 'FT-Data-Ranker-7b-code-model/code-data/data/train_bloom-c4-all-summ.jsonl' #更换为实际存储路径

        # 将每个字典作为一个独立的JSON对象写入文件
        with open(json_file_path, 'w', encoding='utf-8') as file:
            for i in range(len(input_data)):
                temp = {'instruction':input_data[i], 'input':'', 'output':output_data[i]}
                json.dump(temp, file, ensure_ascii=False)
                file.write('\n')  # 在每个JSON对象后添加换行符

if __name__ == "__main__":
    main()
