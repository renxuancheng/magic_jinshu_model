import transformers
from transformers import AutoTokenizer
qwen2_path=f"/mnt/f/Work_data/wsl2/data/tokenizer/qwen2"
pt_data_path=f"/mnt/f/Work_data/wsl2/data/processed_data/01.opennmt_chinese/test.json"
tokenizer=AutoTokenizer.from_pretrained(qwen2_path)
print(tokenizer.encode('刘邦'))
print(tokenizer.special_tokens_map)
dataset=[]
with open(pt_data_path,'r',encoding='utf-8') as f_read:
    dataset=f_read.readlines()

tokenizer=tokenizer.train_new_from_iterator(text_iterator=dataset,vocab_size=80000)
tokenizer.save_pretrained(f"/mnt/f/Work_data/wsl2/data/processed_data/01.opennmt_chinese/token")