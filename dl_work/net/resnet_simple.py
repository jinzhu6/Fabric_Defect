# Author:  Acer Zhang
# Datetime:2019/10/26
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
                 deep_level: int = 2,
                 ipt_size_level: int = 8,
                 classify_level: int = 3):

        """
        初始化SimpleResNet对象
        :param ipt: 网络输入
        :param classify_num: 分类数量
        :param deep_level: 网络深度控制 默认为2 该参数请参考网络配置文档
        :param ipt_size_level: 图片尺寸级别 该参数请参考网络配置文档
        :param classify_level: 分类网络级别 该参数请参考网络配置文档
        """

        assert 100 >= ipt_size_level >= 8, "网络深度不合理，请参考网络文档进行设置"
        assert 200 >= deep_level >= 2, "网络深度不合理，请参考网络文档进行设置"

        self.ipt = ipt
        self.classify_num = classify_num
        self.deep_level = deep_level
        self.ipt_size_level = ipt_size_level
        self.classify_level = classify_level
        self._build_net()

    def req_classify_net(self):
        """
        获取分类网络，以及网络中需要梯度更新的层，其中列表最后一个元素为输出层
        :return: list[layers]
        """
        return self.out_list_classify

    def req_detection_net(self):
        """
        获取目标检测网络
        :return: layer
        """
        return self.out_list_detection

    def req_layer_count(self):
        """
        获取网络层数
        :return: 网络层数
        """
        return self.deep_level * self.ipt_size_level + 1

    def req_detection_layer_size(self):
        """
        获取提供给目标检测特征图尺寸列表
        """
        return [i.shape for i in self.out_list_detection]

    def req_feature_view(self):
        """
        获取该网络特征图
        """
        pass

    def _build_net(self):
        """
        开始组建网络
        """
        tmp = base_layer(
            ipt=self.ipt,
            name="1",
            filter_num=32,
            filter_size=3,
            size_cut=True,
            act='relu',
            depthwise_sc=False)

        out_list_detection = []
        for group_num in range(self.ipt_size_level):
            filters_num = 2 ** (group_num // 2 + 5) if group_num <= 10 else 1024
            tmp = res_block(tmp,
                            name=str(group_num),
                            num_filters=filters_num,
                            deep_level=self.deep_level)

            if group_num % 2 == 1:
                tmp = base_layer(ipt=tmp,
                                 name="res_block_conv_" + str(group_num) + "_cut",
                                 filter_num=2 * filters_num,
                                 filter_size=3,
                                 size_cut=True,
                                 act="relu",
                                 same_padding=False)
                if group_num > 2:
                    out_list_detection.append(tmp)
                else:
                    classify_layer1 = base_layer(ipt=tmp,
                                                 name="classify_layer_conv",
                                                 filter_num=2 * filters_num,
                                                 filter_size=self.classify_level,
                                                 size_cut=True,
                                                 act='relu',
                                                 same_padding=True)
                    classify_layer2 = fluid.layers.fc(input=classify_layer1,
                                                      size=self.classify_level * self.classify_num,
                                                      name="classify_layer_fc",
                                                      act="relu")
                    classify_layer3 = fluid.layers.fc(input=classify_layer2,
                                                      size=2,
                                                      name="classify_layer_out",
                                                      act="softmax")
                    self.out_list_classify = [classify_layer1, classify_layer2, classify_layer3]
        self.out_list_detection = [out_list_detection[0],
                                   out_list_detection[(self.ipt_size_level - 2) // 4],
                                   out_list_detection[-1]]

# Debug

# data = fluid.layers.data(name="debug", shape=[1, 2400, 1200])
# net_obj = SimpleResNet(data, deep_level=2, ipt_size_level=8)
# print("layer_count", net_obj.req_layer_count())
# print("detection_layer_size", net_obj.req_detection_layer_size())
