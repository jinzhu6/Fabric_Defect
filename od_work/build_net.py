# Author:  Acer Zhang
# Datetime:2019/10/27
# Copyright belongs to the author.
# Please indicate the source for reprinting.
from net.resnet_simple import SimpleResNet
import paddle.fluid as fluid
import yaml

# 读取配置文件
with open("./config/net.yaml", "r", encoding="utf-8") as f:
    conf = f.read()
    conf = yaml.load(conf, Loader=yaml.FullLoader)


def build_net(ipt,
              box_ipt_list=None,
              label_list=None,
              mode: str = "Train"):
    """
    构建网络
    :param ipt: 输入张量
    :param box_ipt_list: 标记框列表
    :param label_list: 标签列表
    :param mode: 运行模式 Train、Eval、Infer
    :return: Train模式返回loss 、 Eval模式返回计算后MAP对象、 Infer模式返回NMS结果
    """
    net_obj = SimpleResNet(ipt)
    layer_out = net_obj.req_detection_net()

    mbox_locs, mbox_confs, boxs, vars = fluid.layers.multi_box_head(
        inputs=layer_out,
        image=ipt,
        num_classes=conf["net"]["classify_num"],
        min_ratio=conf["net"]["min_ratio"],
        max_ratio=conf["net"]["max_ratio"],
        aspect_ratios=conf["net"]["aspect_ratios"],
        base_size=conf["net"]["base_size"],
        offset=conf["net"]["offset"],
        flip=True,
        clip=True)
    nms_out = fluid.layers.detection_output(mbox_locs,
                                            mbox_confs,
                                            boxs,
                                            vars,
                                            nms_threshold=conf["net"]["nms_threshold"])  # 非极大值抑制得到的结果
    if mode == "Infer":
        return nms_out
    elif mode == "Train":
        loss = fluid.layers.ssd_loss(location=mbox_locs,
                                     confidence=mbox_confs,
                                     gt_box=box_ipt_list,
                                     gt_label=label_list,
                                     prior_box=boxs,
                                     prior_box_var=vars,
                                     background_label=conf["net"]["background_label"])
        loss = fluid.layers.mean(loss)
        return loss
    elif mode == "Eval":
        map_eval = fluid.metrics.DetectionMAP(nms_out,
                                              label_list,
                                              box_ipt_list,
                                              class_num=conf["net"]["classify_num"],
                                              overlap_threshold=conf["eval"]["overlap_threshold"],
                                              ap_version=conf["eval"]["ap_version"])

        return map_eval
    else:
        print("模式选择有误,请修改build_net中mode参数为‘Train、Eval、Infer’中其一")
        exit(1)



