# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/5/9
@Time    : 13:12
@File    : server_confindence.py
@Function: XX
@Other: XX
"""

import copy
import json
import os
import torch
import time
import socket
import numpy as np
import tqdm
from flask import Flask, request
from torchcrf import CRF
from gevent import pywsgi
from transformers import BertTokenizer
from utils.functions import load_model_and_parallel, get_relation_result_confi
from utils.models import GlobalPointerRe
from utils.tokenizers import sentence_encode


def torch_env():
    """
    测试torch环境是否正确
    :return:
    """
    import torch.backends.cudnn

    print('torch版本:', torch.__version__)  # 查看torch版本
    print('cuda版本:', torch.version.cuda)  # 查看cuda版本
    print('cuda是否可用:', torch.cuda.is_available())  # 查看cuda是否可用
    print('可行的GPU数目:', torch.cuda.device_count())  # 查看可行的GPU数目 1 表示只有一个卡
    print('cudnn版本:', torch.backends.cudnn.version())  # 查看cudnn版本
    print('输出当前设备:', torch.cuda.current_device())  # 输出当前设备（我只有一个GPU为0）
    print('0卡名称:', torch.cuda.get_device_name(0))  # 获取0卡信息
    print('0卡地址:', torch.cuda.device(0))  # <torch.cuda.device object at 0x7fdfb60aa588>
    x = torch.rand(3, 2)
    print(x)  # 输出一个3 x 2 的tenor(张量)


def get_ip_config():
    """
    ip获取
    :return:
    """
    myIp = [item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None) if ':' not in item[4][0]][0]
    return myIp


def encode(texts):
    """

    :param texts: list of str
    :return:
    """
    return sentence_encode(list(texts), args.max_seq_len, tokenizer)


def decode(token_ids, attention_masks, token_type_ids):
    """

    :param token_ids:
    :param attention_masks:
    :param token_type_ids:
    :return:
    """
    results = []
    with torch.no_grad():
        entity, relation = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))

        Rentity, Rrelation = get_relation_result_confi(entity, relation, args.fun_map)
        for item in Rentity:
            item[1][0] = id2entity[item[1][0]]
        for item in Rrelation:
            item[0] = id2relation[item[0]]
        return {'实体': Rentity, '关系': Rrelation}


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


torch_env()
model_name = './checkpoints/chenyang-chinese-bert-wwm-ext-2023-09-18'
args_path = os.path.join(model_name, 'args.json')
model_path = os.path.join(model_name, 'model_best.pt')

entity_labels_path = os.path.join(model_name, 'entity_labels.json')
relation_labels_path = os.path.join(model_name, 'relation_labels.json')
port = 12003
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(entity_labels_path, 'r', encoding='utf-8') as f:
    entity_labels = json.load(f)
with open(relation_labels_path, 'r', encoding='utf-8') as f:
    relation_labels = json.load(f)

# fun_map = {'0-4': 0, '0-5': 1, '0-1': 2, '0-2': 3, '0-3': 4}
id2entity = {k: v for k, v in enumerate(entity_labels)}
entity2id = {v: k for k, v in enumerate(entity_labels)}
id2relation = {k: v for k, v in enumerate(relation_labels)}
relation2id = {v: k for k, v in enumerate(relation_labels)}
args = Dict2Class(**tmp_args)
args.gpu_ids = '0'
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))

model, device = load_model_and_parallel(GlobalPointerRe(args), args.gpu_ids, model_path)

model.eval()
for name, param in model.named_parameters():
    param.requires_grad = False
app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
        # msgs = request.get_json("content")
        msgs = msgs.decode('utf-8')
        token_ids, attention_masks, token_type_ids = encode(msgs)

        partOfResults = decode(token_ids, attention_masks, token_type_ids)

        res = json.dumps(partOfResults, ensure_ascii=False)
        return res
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, threaded=False, debug=False)
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("Starting server in python...")
    print('Service Address : http://' + get_ip_config() + ':' + str(port))
    server.serve_forever()
    print("done!")
    # app.run(host=hostname, port=port, debug=debug)  注释以前的代码
    # manager.run()  # 非开发者模式
