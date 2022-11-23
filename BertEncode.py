import os
import torch
import torch.utils.data as Data
from tqdm import tqdm
from transformers import BertTokenizer
import pickle
from torch.utils.data import DataLoader


class BertGetData:
    """
        数据编码，读取
    """

    def __init__(self, max_seq_length, model_tokenizer, model_name) -> None:
        self.tokenizer = model_tokenizer
        self.max_seq_length = max_seq_length
        self.model_name = model_name

    def encode(self, sentences):
        """
            将输入的句子转成ids
        :param sentences: 句子
        :return: ids
        """
        input_ids = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:self.max_seq_length - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        return input_ids

    def get_input_ids(self, x):
        """
            根据最大的句子长度限制,若超出则取0->max_len长度的ids,若未超出,补0
        :param x:
        :return: ids
        """
        input_ids = self.tokenizer.encode(x)
        if len(input_ids) > self.max_seq_length:
            input_ids = [input_ids[0]] + input_ids[1:self.max_seq_length - 1] + [input_ids[-1]]
        else:
            input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))
        return input_ids

    def read_dataset(self, file_path):
        """
            读取数据,格式为，input_ids, label_ids, input_mask
        :param file_path: txt文件路径
        :return: dataset
        """
        pickle_path = file_path.replace("txt", "pkl")
        pickle_path = pickle_path.split('.')[0] + "_" + self.model_name.lower() + "." + pickle_path.split('.')[1]
        if os.path.exists(pickle_path):
            file = open(pickle_path, mode='rb')
            dataset = pickle.load(file)
            file.close()
        else:
            token_list = []
            label_ids = []
            seq_list = []
            input_mask = []
            with open(file_path, 'r', encoding='UTF-8') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    content, labels = line.split('\t')
                    token_ids = self.get_input_ids(content)
                    token_list.append(token_ids)
                    label_ids.append(list(eval(labels)))
                    seq_list.append(len(token_ids))
                    if len(token_ids) < self.max_seq_length:
                        mask = [1] * len(token_ids) + [0] * (self.max_seq_length - len(token_ids))
                    else:
                        mask = [1] * self.max_seq_length
                    input_mask.append(mask)
            label_ids = torch.tensor(label_ids)
            input_ids = torch.tensor(token_list, dtype=torch.int)
            input_mask = torch.tensor(input_mask, dtype=torch.int)
            seq_len = torch.tensor(seq_list, dtype=torch.int)
            dataset = Data.TensorDataset(input_ids, label_ids, input_mask, seq_len)
            file = open(pickle_path, mode='wb')
            pickle.dump(dataset, file)
            file.close()
        return dataset

    def encode_unknown_len_data(self, sentences):
        """
            处理未知长度的句子，得到ids
        :param sentences:
        :return:
        """
        input_ids = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        return input_ids

    def get_input_ids_unknown(self, x):
        """
            得到未知长度的data的ids
        :param x:
        :return:
        """
        input_ids = self.tokenizer.encode(x)
        return input_ids

    def get_unknow_len_data(self, file_path):
        """
            读取未知最大长度数据
        :param file_path: 文件路径
        :return: max_id:最大长度的id, max_len:最大长度, content_list[max_id]:最大长度的ids
        """
        max_id = 0
        content_list = []
        label_ids = []
        max_len = 0
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, labels = line.split('\t')
                content_list.append(self.get_input_ids_unknown(content))
                label_ids.append(labels)
        for i in range(len(content_list)):
            if len(content_list[i]) > max_len:
                max_id = i
                max_len = len(content_list[i])
        return max_id, max_len, content_list[max_id]

    def get_loader(self, dataset, batch_size, shuffle_dataset=True):
        """
            得到loader
        :param dataset:
        :param batch_size:
        :return:
        """
        # shuffle 是否要打乱顺序
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_dataset, drop_last=False)
        return data_loader


if __name__ == "__main__":
    # 加载数据
    vocab_path = "bert_pretrain/vocab.txt"
    data = BertGetData(vocab_path)
    dataset = data.read_dataset("train.txt")
    # a, b, c = data.get_unknow_len_data("train.txt")
    # print(a)
    # print(b)
    # print(c)
    train_valid = data.get_loader(dataset, 64, shuffle_dataset=False)
