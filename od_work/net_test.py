# Author:  Acer Zhang
# Datetime:2019/11/6
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle.fluid as fluid


def base_layer(ipt,
               name: str,
               filter_num: int,
               filter_size: int = 3,
               act=None,
               size_cut: bool = False,
               same_padding: bool = True):
    """
    基础卷积+BN处理函数
    :param ipt: 输入张量数据
    :param name: 该层命名
    :param filter_num: 卷积核数量
    :param filter_size: 卷积核尺寸
    :param act: 卷积层激活函数
    :param size_cut: 是否剪裁尺寸
    :param same_padding: 是否保持输入输出尺寸
    :return: 处理后张量
    """

    stride = filter_size if size_cut else 1
    padding = (filter_size - 1) // 2 if same_padding else 0

    tmp = fluid.layers.conv2d(
        input=ipt,
        num_filters=filter_num,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        bias_attr=False,
        name="base_conv_" + name)

    tmp = fluid.layers.batch_norm(
        input=tmp,
        act=act,
        name="base_bn_" + name)
    return tmp


def res_block(ipt, name: str, num_filters: int):
    """
    残差模块
    :param ipt: 输入张量
    :param name: 该模块命名
    :param num_filters: 卷积核个数
    :return:
    """
    tmp = base_layer(
        ipt=ipt,
        name="res_block_conv_" + name + "_1",
        filter_num=num_filters,
        filter_size=3,
        size_cut=False,
        act='relu', )
    tmp = base_layer(
        ipt=tmp,
        name="res_block_conv_" + name + "_2",
        filter_num=num_filters,
        filter_size=3,
        size_cut=False)
    tmp = fluid.layers.elementwise_add(x=tmp, y=ipt, act='relu', name="res_block_add_" + name)
    return tmp


class SimpleResNet:
    """
    该残差网络仅适用于纺织物瑕疵识别任务，相对普通残差网络有所改动
    """

    def __init__(self, ipt):
        """
        初始化SimpleResNet对象
        :param ipt: 网络输入
        """
        self.ipt = ipt
        self.size_cut_layers = [1, 3, 5]
        self._build_net()

    def req_classify_net(self):
        return self.out_list_classify

    def req_detection_net(self):
        return self.out_list_detection

    def _build_net(self):
        tmp = base_layer(
            ipt=self.ipt,
            name="1",
            filter_num=64,
            filter_size=3,
            size_cut=True,
            act='relu')

        self.out_list_classify = []
        self.out_list_detection = []
        for group_num in range(6):
            tmp = res_block(tmp, name=str(group_num), num_filters=2 ** (group_num // 2 + 6))
            if group_num % 2 == 1:
                tmp = base_layer(ipt=tmp,
                                 name="res_block_conv_" + str(group_num) + "_3",
                                 filter_num=2 ** (group_num // 2 + 7),
                                 filter_size=3,
                                 size_cut=True,
                                 act='relu', )
                self.out_list_detection.append(tmp)
                print(tmp.shape)


# Debug
# data = fluid.layers.data(name="debug", shape=[3, 1080, 608])
# net_obj = SimpleResNet(data)
# print(net_obj.out_list_detection[0].shape)
