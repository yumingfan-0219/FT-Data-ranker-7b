import json

# 更改为原始文件和目标文件的路径
input_file_path = 'train_bloom-c4.json'
output_file_path = 'train_bloom-c4-temp.json'

# 读取原始文件并解析每一行
json_list = []
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        QA_summ = {'instruction': 'what is the summary of the preceding text?\n', 'input':json_obj['instruction']+json_obj['input']+json_obj['output'], 'output':''}
        json_list.append(QA_summ)

# 将列表写入新文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(json_list, outfile, ensure_ascii=False, indent=4)