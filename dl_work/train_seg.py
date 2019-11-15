# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import time
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image as Image

from tools import img_tool as img_tool
from tools import label_tool as label_tool
from tools import os_tool as os_tool
from build_net import build_net

# 读取配置文件
with open("./config/train_full.yaml", "r", encoding="utf-8") as f:
    conf = f.read()
    conf = dict(yaml.load(conf, Loader=yaml.FullLoader))

dir_path = conf["data_path"] + "/"
train_dir_path = dir_path + "train"
epoch_data_count = 50  # 默认每Epoch有多少图片数量


# Reader
def reader():
    """
    数据读取器
    :return: reader
    """

    def yield_one_data():
        name_list, name_path = os_tool.read_ext_in_dir("./data/train/SegmentationClassPNG",
                                                       ext="png",
                                                       name_none_ext=True)
        for img_id, name in enumerate(name_list):
            im = Image.open('./data/train/JPEGImages/' + name + ".jpg").convert("L")
            im = im.crop((0, 0, 2400, 1200))
            im = im.resize((1600, 800), Image.LANCZOS)
            im = np.array(im).astype("int32")
            im_label = Image.open(name_path[img_id]).convert("L")
            im_label = np.array(im_label)
            im_label = Image.fromarray(label_tool.replace_label(im_label))
            im_label1 = im_label.crop((0, 0, 2400, 1200))
            im_label1 = im_label1.resize((800, 400))
            im_label2 = im_label1.resize((400, 200))
            im_label3 = im_label1.resize((200, 100))

            im_label1 = np.array(im_label1).reshape(1, 400, 800).astype("int32")
            im_mask1 = np.where((im_label1.flatten()) > 0)[0]
            im_label2 = np.array(im_label2).reshape(1, 200, 400).astype("int32")
            im_mask2 = np.where((im_label2.flatten()) > 0)[0]
            im_label3 = np.array(im_label3).reshape(1, 100, 200).astype("int32")
            im_mask3 = np.where((im_label3.flatten()) > 0)[0]
            yield im, im_label1, im_label2, im_label3, im_mask1, im_mask2, im_mask3

    return yield_one_data


# 初始化执行器以及程序
place = fluid.CUDAPlace(0) if conf["use_cuda"] else fluid.CPUPlace()
exe = fluid.Executor(place)
start_prog = fluid.Program()
train_prog = fluid.Program()

with fluid.program_guard(train_prog, start_prog):
    ipt_img = fluid.data(name="ipt_img", shape=[-1, 1, 800, 1600], dtype="float32")

    ipt_label1 = fluid.data(name="ipt_label1", shape=[-1, 1, 400, 800], dtype="int32")
    ipt_label2 = fluid.data(name="ipt_label2", shape=[-1, 1, 200, 400], dtype="int32")
    ipt_label3 = fluid.data(name="ipt_label3", shape=[-1, 1, 100, 200], dtype="int32")
    ipt_mask1 = fluid.data(name="mask1", shape=[-1, 1], dtype="int32", lod_level=1)
    ipt_mask2 = fluid.data(name="mask2", shape=[-1, 1], dtype="int32", lod_level=1)
    ipt_mask3 = fluid.data(name="mask3", shape=[-1, 1], dtype="int32", lod_level=1)
    ipt_label_list = [ipt_label1, ipt_label2, ipt_label3]
    ipt_mask_list = [ipt_mask1, ipt_mask2, ipt_mask3]
    loss, no_grad_set = build_net(ipt_img, 5, ipt_label_list, ipt_mask_list)
    learning_rate = fluid.layers.exponential_decay(learning_rate=conf["e_learning_rate"],
                                                   decay_steps=epoch_data_count // conf["batch_size"],
                                                   decay_rate=0.1,
                                                   staircase=True)
    opt = fluid.optimizer.Adam(learning_rate=learning_rate)
    opt.minimize(loss, no_grad_set=no_grad_set)

# 读取设置

train_reader = paddle.batch(reader=paddle.reader.shuffle(reader(), 500), batch_size=conf["batch_size"])
feeder = fluid.DataFeeder(place=place, feed_list=[ipt_img,
                                                  ipt_label1,
                                                  ipt_label2,
                                                  ipt_label3,
                                                  ipt_mask1,
                                                  ipt_mask2,
                                                  ipt_mask3])

# 训练部分
exe.run(start_prog)
# fluid.io.load_params(executor=exe, dirname="./model/train_od_217", main_program=train_prog)

print("start!")
for epoch in range(conf["epochs"]):
    train_out = []
    start_time = time.time()
    step = 0
    for data_id, data in enumerate(train_reader()):
        train_out = exe.run(program=train_prog,
                            feed=feeder.feed(data),
                            fetch_list=[loss])
        step = data_id
        if data_id == 0:
            cost_time = time.time() - start_time
            print("one data training avg time is:", cost_time / conf["batch_size"])
        print("Epoch:", epoch + 1, "loss:", train_out[0])
    # fluid.io.save_inference_model(dirname="infer_od_" + str(epoch),
    #                               feeded_var_names=['ipt_img'],
    #                               target_vars=[nms],
    #                               executor=exe,
    #                               main_program=train_prog)
    # fluid.io.save_params(executor=exe,
    #                      dirname=conf['save_point_path'] + "/train_od_" + str(epoch),
    #                      main_program=train_prog)
    #
    # map_eval.reset(exe)
    # print("Epoch:", epoch + 1, "loss:", train_out[0], "cur_map:", eval_out[0], "accum_map:", eval_out[1])
