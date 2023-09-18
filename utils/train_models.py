# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 10:04
@File    : train_models.py
@Function: XX
@Other: XX
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pynvml
from tqdm import tqdm
from utils.models import GlobalPointerRe
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, save_model, criterion, get_result, \
    get_relation_result
from utils.adversarial_training import PGD



class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """ y_true ([Tensor]): [..., num_classes]
            y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class MyLossNer(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


class TrainGlobalPointerRe:
    def __init__(self, args, train_loader, dev_loader, entities, relations, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.entities = entities
        self.relations = relations
        self.id2entity = {k: v for k, v in enumerate(entities)}
        self.id2relation = {k: v for k, v in enumerate(relations)}
        self.log = log
        self.criterion = MyLossNer()
        self.model, self.device = load_model_and_parallel(GlobalPointerRe(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)

    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        for epoch in range(1, self.args.train_epochs + 1):
            bar = tqdm(self.train_loader, ncols=160)
            losses = []
            entity_losses = []
            relation_losses = []
            for batch_data in bar:
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                batch_data['entity_labels'] = batch_data['entity_labels'].long()
                batch_data['relation_labels'] = batch_data['relation_labels'].long()
                entity_output, relation_output = self.model(batch_data['token_ids'],
                                                            batch_data['attention_masks'],
                                                            batch_data['token_type_ids'])
                entity_loss = self.criterion(entity_output, batch_data['entity_labels'])
                relation_loss = self.criterion(relation_output, batch_data['relation_labels'])
                loss = (entity_loss + relation_loss) / 2
                losses.append(loss.detach().item())
                entity_losses.append(entity_loss.detach().item())
                relation_losses.append(relation_loss.detach().item())

                bar.set_postfix(loss='%.4f  ' % np.mean(losses) +
                                     '%.4f  ' % np.mean(entity_losses) +
                                     '%.4f  ' % np.mean(relation_losses))
                bar.set_description("[epoch] %s" % str(epoch))
                loss.backward()  # 反向传播 计算当前梯度

                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 如果是最后一次对抗 恢复所有的梯度
                        entity_output_adv, relation_output_adv = self.model(batch_data['token_ids'],
                                                                            batch_data['attention_masks'],
                                                                            batch_data['token_type_ids'])

                        entity_loss_adv = criterion(entity_output_adv, batch_data['entity_labels'])
                        relation_loss_adv = criterion(relation_output_adv, batch_data['relation_labels'])
                        loss_adv = (entity_loss_adv + relation_loss_adv) / 2
                        losses.append(loss_adv.detach().item())
                        entity_losses.append(entity_loss_adv.detach().item())
                        relation_losses.append(relation_loss_adv.detach().item())

                        bar.set_postfix(loss='%.4f  ' % np.mean(losses) +
                                             '%.4f  ' % np.mean(entity_losses) +
                                             '%.4f  ' % np.mean(relation_losses))
                        loss_adv.backward()  # 反向传播 对抗训练的梯度 在最后一次推理的时候 叠加了一次loss
                    pgd.restore()  # 恢复embedding参数

                # 梯度裁剪 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_grad
                # 和clip_grad_value的区别在于 clip_grad_value暴力指定了区间 而clip_grad_norm做范数上的调整
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0

            if epoch > self.args.train_epochs * 0.1:
                f1, precision, recall = self.dev()
                if f1 > best_f1:
                    best_f1 = f1
                    save_model(self.args, self.model)
                self.log.info('[eval] epoch:{} pre={:.6f} rec={:.6f} f1_score={:.6f} best_f1_score={:.6f}'.format(epoch,
                                                                                                                  precision,
                                                                                                                  recall,
                                                                                                                  f1,
                                                                                                                  best_f1))
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.log.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in tqdm(self.dev_loader, leave=False, ncols=80):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)

                entity_output, relation_output = self.model(dev_batch_data['token_ids'],
                                                            dev_batch_data['attention_masks'],
                                                            dev_batch_data['token_type_ids'])

                R_entities = set(get_result(entity_output))
                R_relations = set(get_result(relation_output))

                T_entities = set(get_result(dev_batch_data['entity_labels']))
                T_relations = set(get_result(dev_batch_data['relation_labels']))

                X += len(R_entities & T_entities) + len(R_relations & T_relations)
                Y += len(R_entities) + len(R_relations)
                Z += len(T_entities) + len(T_relations)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            if X == 1e-15 and Y == 1e-15:
                f1, precision, recall = 0., 0., 0.
            return f1, precision, recall

    def test(self, model_path):
        model, device = load_model_and_parallel(GlobalPointerRe(self.args), self.args.gpu_ids, model_path)
        model.eval()
        # 确定有哪些关系
        entity2ids = {v: k for k, v in enumerate(self.entities)}
        relation2ids = {v: k for k, v in enumerate(self.relations)}
        fun_map = self.args.fun_map
        assert len(fun_map) == len(relation2ids)
        XE, YE, ZE = np.full((len(entity2ids),), 1e-15), np.full((len(entity2ids),), 1e-15), np.full((len(entity2ids),),
                                                                                                     1e-15)
        XE_all, YE_all, ZE_all = 1e-15, 1e-15, 1e-15
        XR, YR, ZR = np.full((len(relation2ids),), 1e-15), np.full((len(relation2ids),), 1e-15), np.full(
            (len(relation2ids),), 1e-15)
        XR_all, YR_all, ZR_all = 1e-15, 1e-15, 1e-15

        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader, ncols=80):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)

                entity_output, relation_output = model(dev_batch_data['token_ids'],
                                                       dev_batch_data['attention_masks'],
                                                       dev_batch_data['token_type_ids'])
                Rentity, Rrelation = get_relation_result(entity_output, relation_output, fun_map)
                Rentity, Rrelation = set(Rentity), set(Rrelation)

                Tentity, Trelation = get_relation_result(dev_batch_data['entity_labels'], dev_batch_data['relation_labels'], fun_map)
                Tentity, Trelation = set(Tentity), set(Trelation)

                XE_all += len(Rentity & Tentity)
                YE_all += len(Rentity)
                ZE_all += len(Tentity)

                for item in Rentity & Tentity:
                    XE[item[1]] += 1
                for item in Rentity:
                    YE[item[1]] += 1
                for item in Tentity:
                    ZE[item[1]] += 1

                XR_all += len(Rrelation & Trelation)
                YR_all += len(Rrelation)
                ZR_all += len(Trelation)

                for item in Rrelation & Trelation:
                    XR[item[1]] += 1
                for item in Rrelation:
                    YR[item[1]] += 1
                for item in Trelation:
                    ZR[item[1]] += 1


        f1E, precisionE, recallE = 2 * XE_all / (YE_all + ZE_all), XE_all / YE_all, XE_all / ZE_all
        str_log = '\n' + '实体\t' + 'precision\t' + 'pre_count\t' + 'recall\t' + 'true_count\t' + 'f1-score\n'
        str_log += '' \
                   + '全部实体\t' \
                   + '%.4f' % precisionE + '\t' \
                   + '%.0f' % YE_all + '\t' \
                   + '%.4f' % recallE + '\t' \
                   + '%.0f' % ZE_all + '\t' \
                   + '%.4f' % f1E + '\n'
        f1, precision, recall = 2 * XE / (YE + ZE), XE / YE, XE / ZE
        for entity in self.entities:
            str_log += '' \
                       + entity + '\t' \
                       + '%.4f' % precision[entity2ids[entity]] + '\t' \
                       + '%.0f' % YE[entity2ids[entity]] + '\t' \
                       + '%.4f' % recall[entity2ids[entity]] + '\t' \
                       + '%.0f' % ZE[entity2ids[entity]] + '\t' \
                       + '%.4f' % f1[entity2ids[entity]] + '\n'

        self.log.info(str_log)

        f1R, precisionR, recallR = 2 * XR_all / (YR_all + ZR_all), XR_all / YR_all, XR_all / ZR_all
        str_log = '\n' + '关系\t' + 'precision\t' + 'pre_count\t' + 'recall\t' + 'true_count\t' + 'f1-score\n'
        str_log += '' \
                   + '全部关系\t' \
                   + '%.4f' % precisionR + '\t' \
                   + '%.0f' % YR_all + '\t' \
                   + '%.4f' % recallR + '\t' \
                   + '%.0f' % ZR_all + '\t' \
                   + '%.4f' % f1R + '\n'

        f1, precision, recall = 2 * XR / (YR + ZR), XR / YR, XR / ZR
        for relation in self.relations:
            str_log += '' \
                       + relation + '\t' \
                       + '%.4f' % precision[relation2ids[relation]] + '\t' \
                       + '%.0f' % YR[relation2ids[relation]] + '\t' \
                       + '%.4f' % recall[relation2ids[relation]] + '\t' \
                       + '%.0f' % ZR[relation2ids[relation]] + '\t' \
                       + '%.4f' % f1[relation2ids[relation]] + '\n'
        self.log.info(str_log)


def get_entity_gp_dev(tensors):
    """

    :param tensors: batch * labels_num * max_len * max_len
    :return: list of Tuple: (batch, label, start, end)
    """
    entities = []
    for batch, label, start, end in torch.nonzero(tensors > 0):
        entities.append((batch.item(), label.item(), start.item(), end.item()))
    return entities

