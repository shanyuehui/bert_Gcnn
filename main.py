#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 个人微信 wibrce
# Author 杨博

import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import train
from BertEncode import BertGetData
import sendemail

parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classsification')
parser.add_argument('--model', type=str, default='BruceBert',
                    help='choose a model BruceBert, BruceBertCNN, BruceBertRNN, BruceBertDPCNN,robert_text,robert_cnn')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'WiKi'  # 数据集地址
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    # vocab_path = config.bert_path + "/vocab.txt"
    # data_reader = BertGetData(vocab_path, do_lower_case=True, max_seq_length=config.pad_size)
    data_reader = BertGetData(config.pad_size, config.tokenizer, config.model_name)
    # 训练集
    dataset_train = data_reader.read_dataset(config.train_path)
    train_iter = data_reader.get_loader(dataset_train, config.batch_size, shuffle_dataset=True)
    # 测试和验证集
    dataset_dev = data_reader.read_dataset(config.test_path)
    dataset_test = dataset_dev

    dev_iter = data_reader.get_loader(dataset_dev, config.batch_size, shuffle_dataset=True)
    test_iter = dev_iter
    # train_data, dev_data, test_data = utils.bulid_dataset(config)
    # train_iter = utils.bulid_iterator(train_data, config)
    # dev_iter = utils.bulid_iterator(dev_data, config)
    # test_iter = utils.bulid_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：", time_dif)

    # 模型训练，评估与测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
    # train.test(config, model, test_iter)
    try:
        mail_msg ="""
        <p>尊敬的单岳辉:</p>
        <p>您好，非常抱歉打扰到您，这是一份Python 的邮件测试，看见后可以忽略。</p>
        <p><a>训练结束</a></p>
        """

        # 调用函数（登录密码需要换成你自己的）
        sendemail.mail('799424613@qq.com',
        'spneuyvwswzibecc',
        'shanyuehui@126.com',
        '',
        '', mail_msg)
        print('邮件发送成功！')
    except:
        print('邮件发送失败！')