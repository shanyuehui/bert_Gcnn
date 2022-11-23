#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 个人微信 wibrce
# Author 杨博
import numpy as np
import torch
import torch.nn as nn
from transformers import get_scheduler

import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter


def train(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    start_time = time.time()
    # 启动 BatchNormalization 和 dropout
    model.train()
    # 拿到所有mode种的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=config.learning_rate
                      # warmup=0.05,
                      # t_total=len(train_iter) * config.num_epochs
                      )

    # 学习率衰减
    num_training_steps = len(train_iter) * config.num_epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0.05, num_training_steps=num_training_steps
    )
    loss_function = nn.BCELoss()

    total_batch = 0  # 记录进行多少batch
    dev_best_loss = float('inf')  # 记录校验集合最好的loss
    last_imporve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练
    model.train()
    writer = SummaryWriter(config.tensorboard_path)

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}'.format(epoch + 1, config.num_epochs))
        for i, content in enumerate(train_iter):
            input_id = content[0].to(config.device)
            labels = content[1].float().to(config.device)  # 根据损失函数确定数据类型
            masks = content[2].to(config.device)
            seq_len = content[3].to(config.device)
            trains = (input_id, seq_len, masks)
            outputs = model(trains)
            model.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward(retain_graph=False)
            optimizer.step()
            lr_scheduler.step()
            if total_batch % 100 == 0:  # 每多少次输出在训练集和校验集上的效果
                # true = labels.data.cpu().numpy()
                true = labels.data.cpu()
                # predit = torch.max(outputs.data, 1)[1].cpu()
                predit = np.where(outputs.data.cpu() > 0.5, 1, 0)  # 大于阈值0.5转为1进行评估
                train_acc = metrics.accuracy_score(true, predit)
                # train_micro = metrics.f1_score(true, predit, average='micro')
                # train_macro = metrics.f1_score(true, predit, average='macro')
                # train_acc = np.mean(np.equal(true, predit))
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # dev_acc, dev_loss, dev_micro, dev_macro = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_imporve = total_batch
                else:
                    imporve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, imporve))

                writer.add_scalar("Train Loss", loss.item(), total_batch)
                writer.add_scalar("Train Acc", train_acc, total_batch)
                writer.add_scalar("Val Loss", dev_loss, total_batch)
                writer.add_scalar("Val Acc", dev_acc, total_batch)

                model.train()
            total_batch = total_batch + 1
            if total_batch - last_imporve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

        if flag:
            break
    test(config, model, test_iter)
    writer.close()


def evaluate(config, model, dev_iter, test=False):
    """

    :param config:
    :param model:
    :param dev_iter:
    :return:
    """
    model.eval()
    loss_total = 0
    predict_list = []
    labels_list = []
    # predict_all = np.array([], dtype=int)
    # labels_all = np.array([], dtype=int)
    loss_function = nn.BCELoss()
    with torch.no_grad():
        for i, content in enumerate(dev_iter):
            input_id = content[0].to(config.device)
            labels = content[1].float().to(config.device)  # 根据损失函数确定数据类型
            masks = content[2].to(config.device)
            seq_len = content[3].to(config.device)
            texts = (input_id, seq_len, masks)
            outputs = model(texts)

            loss = loss_function(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = np.where(outputs.data.cpu() > 0.5, 1, 0)  # 大于阈值0.5转为1进行评估
            for tmp in labels:
                predict_list.append(tmp)
            for tmp in predict:
                labels_list.append(tmp)

            # labels_all = np.append(labels_all, labels)
            # predict_all = np.append(predict_all, predict)
    labels_all = np.asarray(labels_list, dtype=int)
    predict_all = np.asarray(predict_list, dtype=int)
    acc = metrics.accuracy_score(labels_all, predict_all)
    # f1_micro = metrics.f1_score(labels_all, predict_all, average='micro')
    # f1_macro = metrics.f1_score(labels_all, predict_all, average='macro')
    if test:
        f1_micro = metrics.f1_score(labels_all, predict_all, average='micro')
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all.argmax(axis=1), predict_all.argmax(axis=1))
        # f1_macro = metrics.f1_score(labels_all, predict_all, average='macro')
        f1_macro_real = 0.0
        k = 0
        for i in range(config.num_classes):
            y_true = labels_all[:, i]
            y_pred = predict_all[:, i]
            tmp_f1 = metrics.f1_score(y_true, y_pred)
            if tmp_f1 > 0:
                k += 1
                f1_macro_real += tmp_f1
        if f1_macro_real != 0:
            f1_macro = f1_macro_real / k
        else:
            f1_macro = 0
        # return acc, loss_total / len(dev_iter), report, confusion
        precision = metrics.precision_score(labels_all, predict_all, average='samples')
        recall = metrics.recall_score(labels_all, predict_all, average='samples')
        return acc, loss_total / len(dev_iter), f1_micro, f1_macro, precision, recall, report, confusion

    return acc, loss_total / len(dev_iter)
    # return acc, loss_total / len(dev_iter), f1_micro, f1_macro


def test(config, model, test_iter):
    """
    模型测试
    :param config:
    :param model:
    :param test_iter:
    :return:
    """
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    test_acc, test_loss, test_micro, test_macro, test_precision, test_recall, test_report, test_confusion = evaluate(
        config, model, test_iter,
        test=True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print('Test micro:' + str(test_micro) + ',macro:' + str(test_macro))
    print('Test precision:' + str(test_precision) + ',recall:' + str(test_recall))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：", time_dif)

def train_typefushion(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    start_time = time.time()
    # 启动 BatchNormalization 和 dropout
    model.train()
    # 拿到所有mode种的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=config.learning_rate
                      # warmup=0.05,
                      # t_total=len(train_iter) * config.num_epochs
                      )

    # 学习率衰减
    num_training_steps = len(train_iter) * config.num_epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0.05, num_training_steps=num_training_steps
    )
    loss_function = nn.BCELoss()

    total_batch = 0  # 记录进行多少batch
    dev_best_loss = float('inf')  # 记录校验集合最好的loss
    last_imporve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练
    model.train()
    writer = SummaryWriter(config.tensorboard_path)

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, content in enumerate(train_iter):
            input_id = content[0].to(config.device)
            labels = content[1].float().to(config.device)  # 根据损失函数确定数据类型
            masks = content[2].to(config.device)
            seq_len = content[3].to(config.device)
            trains = (input_id, seq_len, masks,labels)
            outputs = model(trains)
            model.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward(retain_graph=False)
            optimizer.step()
            lr_scheduler.step()
            if total_batch % 100 == 0:  # 每多少次输出在训练集和校验集上的效果
                # true = labels.data.cpu().numpy()
                true = labels.data.cpu()
                # predit = torch.max(outputs.data, 1)[1].cpu()
                predit = np.where(outputs.data.cpu() > 0.5, 1, 0)  # 大于阈值0.5转为1进行评估
                train_acc = metrics.accuracy_score(true, predit)
                # train_micro = metrics.f1_score(true, predit, average='micro')
                # train_macro = metrics.f1_score(true, predit, average='macro')
                # train_acc = np.mean(np.equal(true, predit))
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # dev_acc, dev_loss, dev_micro, dev_macro = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_imporve = total_batch
                else:
                    imporve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, imporve))

                writer.add_scalar("Train Loss", loss.item(), total_batch)
                writer.add_scalar("Train Acc", train_acc, total_batch)
                writer.add_scalar("Val Loss", dev_loss, total_batch)
                writer.add_scalar("Val Acc", dev_acc, total_batch)

                model.train()
            total_batch = total_batch + 1
            if total_batch - last_imporve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

        if flag:
            break
    test(config, model, test_iter)
    writer.close()