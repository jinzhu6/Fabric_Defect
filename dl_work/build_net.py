# Author:  Acer Zhang
# Datetime:2019/10/27
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import numpy as np
import paddle.fluid as fluid
from net.seg_net import SimpleResNet

with open("./config/net.yaml", "r", encoding="utf-8") as f:
    conf = f.read()

conf = dict(yaml.load(conf, Loader=yaml.FullLoader))


def build_net(ipt,
              box_list=None,
              label_list=None,
              mode: str = "Train1"):
    """
    构建网络
    :param ipt: 输入张量
    :param box_list: 标记框列表
    :param label_list: 标签列表
    :param mode: 运行模式 Train_1、Eval1、TandV2、Infer
    :return: Train模式返回loss 、 Eval模式返回计算后MAP对象、 Infer模式返回NMS结果
    """
    # 读取配置文件

    net_obj = SimpleResNet(ipt=ipt, classify_num=4)

    seg_layers = net_obj.req_seg_layers()
    def create_loss():

        return
