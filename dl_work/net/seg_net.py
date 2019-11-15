# Author:  Acer Zhang
# Datetime:2019/11/13
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


def base_layer(ipt,
               name: str,
               filter_num: int,
               filter_size: int = 3,
               act=None,
               size_cut: bool = False,
               same_padding: bool = True,
               depthwise_sc: bool = True):
    """
    基础卷积+BN处理函数
    :param ipt: 输入张量数据
    :param name: 该层命名
    :param filter_num: 卷积核数量
    :param filter_size: 卷积核尺寸
    :param groups: 卷积分组数
    :param act: 卷积层激活函数
    :param size_cut: 是否剪裁尺寸
    :param same_padding: 是否保持输入输出尺寸
    :param depthwise_sc: 是否深度可分离卷积，若为是则忽略卷积核数量这个参数
    :return: 处理后张量
    """
    parameter_attr = ParamAttr(learning_rate=0.01, initializer=MSRA())
    stride = filter_size - 1 if size_cut else 1
    padding = (filter_size - 1) // 2 if same_padding else 0
    tmp = fluid.layers.conv2d(
        input=ipt,
        num_filters=ipt.shape[-3] if depthwise_sc else filter_num,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        bias_attr=False,
        name="base_conv_" + name,
        param_attr=parameter_attr)
    if depthwise_sc:
        tmp = fluid.layers.conv2d(
            input=tmp,
            num_filters=filter_num,
            filter_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
            name="base_conv_dpsc_" + name,
            param_attr=parameter_attr)

    tmp = fluid.layers.batch_norm(
        input=tmp,
        act=act,
        name="base_bn_" + name)
    return tmp


def res_block(ipt, name: str, num_filters: int, deep_level: int = 2):
    """
    残差模块
    :param ipt: 输入张量
    :param name: 该模块命名
    :param num_filters: 卷积核个数
    :param deep_level:网络深度控制 默认为1
    :return:layer
    """
    tmp = ipt
    for _ in range(deep_level):
        tmp = base_layer(
            ipt=tmp,
            name="res_block_conv_" + name + "_" + str(1 + deep_level),
            filter_num=num_filters,
            filter_size=3,
            size_cut=False,
            act='relu', )
    tmp = fluid.layers.elementwise_add(x=tmp, y=ipt, act='relu', name="res_block_add_" + name)

    return tmp


class SimpleResNet:
    """
    该残差网络仅适用于纺织物瑕疵识别任务，相对普通残差网络有所改动
    """

    def __init__(self,
                 ipt,
                 classify_num: int = 10,
                 classify_level: int = 3,
                 conv_level: list = None):
        """
        初始化SimpleResNet对象
        :param ipt: 网络输入
        :param classify_num: 分类数量
        :param deep_level: 网络深度控制 默认为2 该参数请参考网络配置文档
        :param ipt_size_level: 图片尺寸级别 该参数请参考网络配置文档
        :param classify_level: 分类网络级别 该参数请参考网络配置文档
        """

        self.conv_level = conv_level
        if self.conv_level is None:
            self.conv_level = [2] + [1]
        self.ipt = ipt
        self.classify_num = classify_num
        self.classify_level = classify_level
        self._build_net()

    def req_seg_layers(self):
        layers_list = [base_layer(ipt=tmp,
                                  name="cut_" + str(cut_id),
                                  filter_num=self.classify_num,
                                  filter_size=1,
                                  depthwise_sc=False,
                                  act="softmax")
                       for cut_id, tmp in enumerate(self.layers_list)]
        return layers_list

    def _build_net(self):
        tmp = base_layer(
            ipt=self.ipt,
            name="ipt_conv",
            filter_num=16,
            filter_size=3,
            act='relu',
            depthwise_sc=False)
        # tmp = base_layer(
        #     ipt=tmp,
        #     name="ipt_conv",
        #     filter_num=32,
        #     filter_size=3,
        #     act='relu',
        #     depthwise_sc=False)
        now_filter_num = 16
        layers_list = []
        for cut_id, cut_level in enumerate(self.conv_level):
            for id_, layer_num in enumerate(range(cut_level)):
                tmp = res_block(ipt=tmp,
                                name=str(cut_id) + "_" + str(layer_num) + "_",
                                num_filters=now_filter_num)
            if now_filter_num < 256:
                now_filter_num *= 2
            tmp = base_layer(ipt=tmp,
                             name="cut_" + str(cut_id),
                             filter_num=now_filter_num,
                             filter_size=3,
                             size_cut=True,
                             depthwise_sc=False,
                             act="relu")
            layers_list.append(tmp)

        self.layers_list = layers_list

# # Debug
# data = fluid.layers.data(name="debug", shape=[1, 2400, 1200])
# net = SimpleResNet(data)
