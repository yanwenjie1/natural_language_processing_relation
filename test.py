# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : test.py
@Function: XX
@Other: XX
"""
import json
import requests
from tqdm import tqdm
sess_web = requests.Session()


def server_test(sen):
    # noinspection PyBroadException
    try:
        results = sess_web.post(url=url, data=sen.encode("utf-8")).text
    except Exception as e:
        results = str(e)
    return results


if __name__ == '__main__':
    url = 'http://10.17.107.66:12003/prediction'
    text = '选举廖俊雄先生、黄贤畅先生、孟学女士为薪酬与考核委员会委员，其中廖俊雄先生为主任委员。'
    # text = '于2022年5月27日受到中国银行保险监督管理委员会宁波监管局的行政处罚（甬银保监罚决字(2022)44号）；经查，宁波银行存在非标投资业务管理不审慎、理财业务管理不规范、主承销债券管控不到位、违规办理衍生产品交易业务、信用证议付资金用于购买本行理财、违规办理委托贷款业务、非银融资业务开展不规范、内控管理不到位、数据治理存在欠缺等违规行为，对其处以罚款290万元。于2022年9月8日受到中国银行保险监督管理委员会宁波监管局的行政处罚（甬银保监罚决字(2022)60号）；经查，宁波银行存在柜面业务内控管理不到位的违规行为，对其处以罚款25万元。于2023年1月9日受到中国银行保险监督管理委员会宁波监管局的行政处罚（甬银保监罚决字(2023)1号）；经查，宁波银行存在违规开展异地互联网贷款业务、互联网贷款业务整改不到位、资信见证业务开展不审慎、资信见证业务整改不到位、贷款“三查”不尽职、新产品管理不严格等问题，对其处以220万元罚款。'

    results = server_test(json.dumps(text, ensure_ascii=False))
    # results = server_test(json.dumps(text, ensure_ascii=False))
    print(results)
    # print(json.loads(results))
    # for i in json.loads(results):
    #     print(i)
    # for i in tqdm(range(10000)):
    #    _ = server_test(json.dumps(tests, ensure_ascii=False))
       # _ = server_test(json.dumps(test, ensure_ascii=False), "http://10.17.107.66:8019/prediction")

    # print(json.loads(aa))
