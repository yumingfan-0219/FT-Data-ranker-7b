1.对于英文数据集，我们使用FT-Data-Ranker-7b-code-model/code-data/competition_kit/data-juicer/configs/using_configs中的bloom、c4三个yaml文件逐次进行处理
即，把raw_data_en先经过bloom-oscar.yaml处理，得到结果，再把结果用redpajama-c4-refine.yaml进行处理，得到最终的英文数据，train-bloom-c4.json(在第二阶段，这个结果只是一个中间结果)

1.1观察测试数据，我们发现模型在摘要任务上的性能较差，因此尝试将数据的任务范式进行转换，将问+答数据整合起来，并使用qwen-14b开源模型对train-bloom-c4.json数据进行推理，
并增加一些关于摘要任务的prompt。在处理时，我们遵循官方的要求，将top-p和temp参数都设置为1。代码可参见FT-Data-Ranker-7b-code-model/code-data/competition_kit/qwen-infer.py

1.2通过上述操作，我们额外生成了1w条左右新的关于文本摘要人物的问答对数据，得到competition_kit/data/raw_data/train_bloom-c4-summ.jsonl数据
1.3我们利用train_bloom-c4-summ.jsonl这一新的数据和原来的英文、中文数据一起生成新的训练数据，train_bloom-c4-qwen14b-1w-summ.json。（在第二阶段，这个结果是最终的训练数据）
顾名思义，该数据使用了1w条新生成的【摘要】类型数据，大概使用了1.5M左右的token，具体参见FT-Data-Ranker-7b-code-model/code-data/competition_kit/lm-training/get_train_dataset_7b.py

#以下步骤和阶段1相同
2.对于中文数据集，我们使用上述文件夹中的alpaca文件进行处理
3.在数据采样阶段，我们设计了一些针对关键词的硬匹配采样，具体可以参见competition_kit/lm-training/get_train_dataset_7b.py，同时保证了tokens_nums维持一致，最终得到训练集
4.训练data参见data文件夹
5.我们没有对训练参数进行变更
6.在推理时我们采用fp32的精度

