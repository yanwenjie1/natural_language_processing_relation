# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : train.py
@Function: XX
@Other: XX
"""
import datetime
import os
import shutil
import logging
import torch
from utils.functions import set_seed, set_logger, save_json, reset_console
from utils.train_models import TrainGlobalPointerRe
import config
import json
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import pickle

args = config.Args().get_parser()
logger = logging.getLogger(__name__)


class GPDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.token_ids = [example.token_ids.long() for example in features]
        self.attention_masks = [example.attention_masks.byte() for example in features]
        self.token_type_ids = [example.token_type_ids.long() for example in features]
        self.entity_labels = [example.entity_labels for example in features]
        self.relation_labels = [example.relation_labels for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'entity_labels': self.entity_labels[index],
                'relation_labels': self.relation_labels[index]
                }

        return data



if __name__ == '__main__':
    args.data_name = os.path.basename(os.path.abspath(args.data_dir))
    args.model_name = os.path.basename(os.path.abspath(args.bert_dir))
    args.save_path = os.path.join('./checkpoints',
                                  args.data_name + '-' + args.model_name
                                  + '-' + str(datetime.date.today()))
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)
    # 复制对应的labels文件
    shutil.copy(os.path.join(args.data_dir, 'entity_labels.json'),
                os.path.join(args.save_path, 'entity_labels.json'))
    shutil.copy(os.path.join(args.data_dir, 'relation_labels.json'),
                os.path.join(args.save_path, 'relation_labels.json'))
    shutil.copy(os.path.join(args.data_dir, 'fun_map.json'),
                os.path.join(args.save_path, 'fun_map.json'))
    set_logger(os.path.join(args.save_path, 'log.txt'))
    torch.set_float32_matmul_precision('high')

    if args.data_name == "chenyang":
        # set_seed(args.seed)
        args.batch_size = 16
        args.train_epochs = 20
        args.use_advert_train = True
        args.max_seq_len = 128

    if args.data_name == "yanhaitao":
        # set_seed(args.seed)
        args.batch_size = 8
        args.train_epochs = 20
        args.use_advert_train = False
        args.max_seq_len = 512

    with open(os.path.join(args.data_dir, 'entity_labels.json'), 'r', encoding='utf-8') as f:
        entities = json.load(f)
    with open(os.path.join(args.data_dir, 'relation_labels.json'), 'r', encoding='utf-8') as f:
        relations = json.load(f)
    with open(os.path.join(args.data_dir, 'fun_map.json'), 'r', encoding='utf-8') as f:
        fun_map = json.load(f)

    args.num_entity_tags = len(entities)
    args.num_relation_tags = len(relations)
    args.fun_map = fun_map

    reset_console(args)
    save_json(args.save_path, vars(args), 'args')

    with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
        train_features = pickle.load(f)
    train_dataset = GPDataset(train_features)
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=False,
                              num_workers=0)
    with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
        dev_features = pickle.load(f)
    dev_dataset = GPDataset(dev_features)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            sampler=dev_sampler,
                            shuffle=False,
                            num_workers=0)

    GpForSequence = TrainGlobalPointerRe(args, train_loader, dev_loader, entities, relations, logger)
    GpForSequence.train()
