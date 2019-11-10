# Author:  Acer Zhang
# Datetime:2019/10/27
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import numpy as np
import paddle.fluid as fluid
from net.resnet_simple import SimpleResNet

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

    net_obj = SimpleResNet(ipt,
                           classify_num=conf["classify_num"],
                           deep_level=conf["deep_level"],
                           ipt_size_level=conf["ipt_size_level"],
                           classify_level=conf["classify_level"])

    def get_od_out():
        od_layer_out = net_obj.req_detection_net()
        mbox_locs, mbox_confs, boxs, bvars = fluid.layers.multi_box_head(
            inputs=od_layer_out,
            image=ipt,
            num_classes=conf["classify_num"],
            min_ratio=conf["min_ratio"],
            max_ratio=conf["max_ratio"],
            aspect_ratios=conf["aspect_ratios"],
            base_size=conf["base_size"],
            offset=conf["offset"],
            flip=True,
            clip=True)
        nms_out = fluid.layers.detection_output(mbox_locs,
                                                mbox_confs,
                                                boxs,
                                                bvars,
                                                nms_threshold=conf["nms_threshold"])  # 非极大值抑制得到的结果
        return mbox_locs, mbox_confs, boxs, bvars, nms_out

    if mode == "Infer":
        return get_od_out()[-1]
    elif mode == "TandV1":
        mbox_locs, mbox_confs, boxs, bvars, nms_out = get_od_out()
        loss = fluid.layers.ssd_loss(location=mbox_locs,
                                     confidence=mbox_confs,
                                     gt_box=box_list,
                                     gt_label=label_list,
                                     prior_box=boxs,
                                     prior_box_var=bvars,
                                     background_label=conf["background_label"])
        loss = fluid.layers.mean(loss)

        map_eval = fluid.metrics.DetectionMAP(nms_out,
                                              label_list,
                                              box_list,
                                              class_num=conf["classify_num"],
                                              overlap_threshold=conf["overlap_threshold"],
                                              ap_version=conf["ap_version"])
        return loss, map_eval

    elif mode == "TandV2":
        classify_all_out = net_obj.req_classify_net()
        return classify_all_out
    else:
        print("模式选择有误,请修改build_net中mode参数为‘TandV1、TandV2、Infer’中其一")
        exit(1)
