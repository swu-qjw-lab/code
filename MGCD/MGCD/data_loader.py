import json
import torch
from dataset_config import get_dataset_config
DATA_CFG = get_dataset_config()


class TrainDataLoader(object):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.ptr = 0
        self.data = []

        data_file = DATA_CFG["train_file"]
        config_file = DATA_CFG["config_file"]

        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

        with open(config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')

        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def next_batch(self):
        if self.is_end():
            return None, None, None, None

        end_idx = min(self.ptr + self.batch_size, self.data_len)
        batch_data = self.data[self.ptr:end_idx]
        self.ptr = end_idx

        input_stu_ids = []
        input_exer_ids = []
        input_knowledge_embs = []
        ys = []

        for log in batch_data:
            knowledge_emb = [0.] * self.knowledge_dim

            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0

            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowledge_embs.append(knowledge_emb)
            ys.append(log['score'])

        return (
            torch.LongTensor(input_stu_ids),
            torch.LongTensor(input_exer_ids),
            torch.Tensor(input_knowledge_embs),
            torch.LongTensor(ys)
        )

    def is_end(self):
        return self.ptr >= self.data_len

    def reset(self):
        self.ptr = 0
# data_loader.py
class ValTestDataLoader(object):
    def __init__(self, d_type='predict', batch_size=256,local_map=None):
        self.ptr = 0
        self.data = []
        self.d_type = d_type
        self.batch_size = batch_size  # 添加批次大小参数

        if d_type != "predict":
            raise ValueError(f"Unsupported d_type: {d_type}")
        data_file = DATA_CFG["test_file"]
        config_file = DATA_CFG["config_file"]
        with open(data_file, encoding='utf8') as i_f:
            json_data = json.load(i_f)
        self.flat_data = []
        for user_data in json_data:
            user_id = user_data['user_id']
            for log in user_data['logs']:
                self.flat_data.append({
                    'user_id': user_id,
                    'exer_id': log['exer_id'],
                    'knowledge_code': log['knowledge_code'],
                    'score': log['score']
                })
        
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)
        
        self.data_len = len(self.flat_data)  # 数据总长度
    def __len__(self):
        return self.data_len
    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        
        end_idx = min(self.ptr + self.batch_size, self.data_len)
        batch_data = self.flat_data[self.ptr:end_idx]
        self.ptr = end_idx
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for item in batch_data:
            knowledge_emb = [0.] * self.knowledge_dim
            for code in item['knowledge_code']:
                knowledge_emb[code - 1] = 1.0
            input_stu_ids.append(item['user_id'] - 1)
            input_exer_ids.append(item['exer_id'] - 1)
            input_knowledge_embs.append(knowledge_emb)
            ys.append(item['score'])
        
        return (
            torch.LongTensor(input_stu_ids),
            torch.LongTensor(input_exer_ids),
            torch.Tensor(input_knowledge_embs),
            torch.LongTensor(ys)
        )

    def is_end(self):
        return self.ptr >= self.data_len

    def reset(self):
        self.ptr = 0
