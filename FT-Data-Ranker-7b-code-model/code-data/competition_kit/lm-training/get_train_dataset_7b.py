import os
from transformers import AutoTokenizer 
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import time
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKEN_NUMS = 10000000
RATIO = 0.65
EN_DATA_DIR = "competition_kit/data/raw_data/raw_data_en-bloom-c4.jsonl"
EN_DATA_DIR_SUMM = "competition_kit/data/raw_data/train_bloom-c4-all-summ.jsonl"
ZH_DATA_DIR = "competition_kit/data/raw_data/raw_data_zh_bloom.jsonl"

OUTPUT_FILES = "competition_kit/data/train_data/train_bloom-c4-qwen14b-1w-summ.json" 


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Input:\n\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Response:"
    ),
}

def get_token_count(sample):
    instruction_len = len(enc.tokenize(sample['instruction']))
    input_len = len(enc.tokenize(sample['input']))
    output_len = len(enc.tokenize(sample['output']))
    total_prompt_input_len = instruction_len + input_len + prompt_input_len
    return total_prompt_input_len

def findAllFile(base):
    files = []
    if os.path.isfile(base):
        return [base]
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('jsonl'):
                files.append(os.path.join(root, f))
    return files
    


enc = AutoTokenizer.from_pretrained("competition_kit/models/Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
enc.model_max_length = 1000000000000000019884624838656
prompt_input_len = len(enc.tokenize(PROMPT_DICT["prompt_input"]))

#en_files = findAllFile(EN_DATA_DIR)
en_files = EN_DATA_DIR
ds_en = load_dataset('json', data_files=en_files, split='train').shuffle(seed=123).select_columns(['instruction', 'input', 'output'])
ds_en_summ = load_dataset('json', data_files=EN_DATA_DIR_SUMM, split='train').shuffle(seed=123).select_columns(['instruction', 'input', 'output'])

en_token_nums = TOKEN_NUMS * RATIO if ZH_DATA_DIR else TOKEN_NUMS
zh_token_nums = TOKEN_NUMS - en_token_nums
selected_elements = []
index_in_summ     = []
count = 0




count = 0
# 将 selected_elements 转换为一个数据集
selected_dataset = Dataset.from_dict({key: [d[key] for d in selected_elements] for key in selected_elements[0]}).select_columns(['instruction', 'input', 'output'])


for i in tqdm(range(len(ds_en_summ))):
    en_token_nums -= get_token_count(ds_en_summ[i])
    if i >= 0.5*len(ds_en_summ):
        break


ds_en_summ = ds_en_summ.select(range(i+1)).select_columns(['instruction', 'input', 'output'])

for i in tqdm(range(len(ds_en))):
    count += get_token_count(ds_en[i])
    print('en num_tokens', i, count)
    if count >= en_token_nums:
        break

ds_en = ds_en.select(range(i+1)).select_columns(['instruction', 'input', 'output'])

ds_en = concatenate_datasets([ds_en, selected_dataset]).shuffle(seed=123)
ds_en = concatenate_datasets([ds_en, ds_en_summ]).shuffle(seed=123)

if ZH_DATA_DIR:
    zh_files = ZH_DATA_DIR
    ds_zh = load_dataset('json', data_files=zh_files, split='train').shuffle(seed=123)
    count = 0
    for i in range(len(ds_zh)):
        count += get_token_count(ds_zh[i])
        print('zh num_tokens', i, count)
        if count >= zh_token_nums:
            break
    ds_zh = ds_zh.select(range(i+1)).select_columns(['instruction', 'input', 'output'])
    ds = concatenate_datasets([ds_en, ds_zh]).shuffle(seed=123)
    ds.to_json(OUTPUT_FILES, force_ascii=False)
else:
    ds_en.to_json(OUTPUT_FILES, force_ascii=False)

print(en_token_nums, zh_token_nums)