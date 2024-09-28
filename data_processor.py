import os
import random
import json
random.seed(114514)
class DataProcessor:
    '''
    数据预处理类
    '''
    def __init__(self,in_data_path,ou_data_path,raw_data_name,train_split_ratio=0.95,test_split_ratio=0.025):
        self.in_data_path=in_data_path
        self.ou_data_path=ou_data_path
        self.train_split_ratio=train_split_ratio
        self.test_split_ratio=test_split_ratio
        self.raw_data_name=raw_data_name
    def convert_pt_data(self):
        print(f"process pretrain raw data name: {self.raw_data_name}")
        data_json_list=[]
        if self.raw_data_name=='opennmt_chinese':
            
            paths = os.walk(self.in_data_path)
            for path, _, file_lst in paths:
                for file_name in file_lst:
                    full_file_name = os.path.join(path, file_name)
                    if '古文原文' in full_file_name or ('双语数据' in full_file_name and 'target' in full_file_name):
                        print(full_file_name)
                        with open(full_file_name,'r',encoding='utf-8') as f_read:
                            ancient_chinese_data_json={}
                            ancient_chinese_data_json['id']=full_file_name.replace(self.in_data_path,"").replace('/','_')
                            ancient_chinese_data_json['text']=f_read.read()
                            data_json_list.append(ancient_chinese_data_json)
            self.save_convert_data(data_json_list)
    def save_convert_data(self,data_json_list):
        random.shuffle(data_json_list)

        train_pt_data_list=data_json_list[0:int(len(data_json_list)*self.train_split_ratio)]
        test_pt_data_list=data_json_list[int(len(data_json_list)*self.train_split_ratio):
                                            int(len(data_json_list)*(self.train_split_ratio+self.test_split_ratio))]
        validate_pt_data_list=data_json_list[int(len(data_json_list)*(self.train_split_ratio+self.test_split_ratio)):]

        save_json={f"{self.ou_data_path}/train.json":train_pt_data_list,
                   f"{self.ou_data_path}/test.json":test_pt_data_list,
                   f"{self.ou_data_path}/validate.json":validate_pt_data_list}
        for path,data in save_json.items():
            if len(data)==0:
                continue
            else:
                with open(path,'w',encoding='utf-8') as f_write:
                    for line in data:
                        f_write.write(json.dumps(line,ensure_ascii=False)+'\n')

    def convert_sft_data(self):
        pass

        
if __name__ =="__main__":
    print()
    in_root_path=f"/mnt/f/Work_data/wsl2/data/raw_data/01.opennmt_chinese"
    ou_root_path=f"/mnt/f/Work_data/wsl2/data/processed_data/01.opennmt_chinese"
    raw_data_name="opennmt_chinese"
    data_processor=DataProcessor(in_root_path,ou_root_path,raw_data_name)
    data_processor.convert_pt_data()