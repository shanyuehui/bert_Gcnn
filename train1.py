
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from transformers.optimization import AdamW, get_scheduler


def train(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练方法
    :param config:模型参数
    :param model:模型
    :param train_iter:训练集
    :param dev_iter:验证集
    :param test_iter:测试集
    :return:
    """
    start_time = time.time()
    #启动 BatchNormalization 和 dropout
    model.train()
    #拿到所有model的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数，参数衰减主要是过拟合状态的减少
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer  if not any( nd in n for nd in no_decay)], 'weight_decay':0.01},#需要衰减的参数
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}#不需要衰减的参数
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters,
                         lr=config.learning_rate
                         #warmup=0.05,
                         #t_total=len(train_iter) * config.num_epochs
                    )
    #学习率衰减
    num_training_steps=len(train_iter)*config.num_epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0.05, num_training_steps=num_training_steps
    )
    total_batch = 0 #记录进行多少batch
    dev_best_loss = float('inf') #记录校验集合最好的loss
    last_imporve = 0 #记录上次校验集loss下降的batch数
    flag = False #记录是否很久没有效果提升，停止训练
    model.train()
    writer=SummaryWriter(config.tensorboard_path)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains).to(config.device)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward(retain_graph=False)
            optimizer.step()

            if total_batch % 100 == 0: #每多少次输出在训练集和校验集上的效果


                true = labels.data.cpu()
                predit = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predit)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_imporve = total_batch
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                writer.add_scalar("Train Loss", loss.item(), total_batch)
                writer.add_scalar("Train Acc", train_acc, total_batch)
                writer.add_scalar("Val Loss", dev_loss, total_batch)
                writer.add_scalar("Val Acc",dev_acc , total_batch)


                model.train()
            total_batch = total_batch + 1

            if total_batch - last_imporve > config.require_improvement:#early-stop
                #在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

        if flag:
            break
    test(config, model, test_iter)
    writer.close()
def evaluate(config, model, dev_iter, test=False):
    """

    :param config:定义config参数类
    :param model:模型
    :param dev_iter:验证数据集
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


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

    test_acc, test_loss ,test_report, test_confusion = evaluate(config, model, test_iter, test = True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'

    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：",time_dif)


















