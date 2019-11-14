# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import time
import paddle.fluid as fluid
import paddle
import numpy as np
import PIL.Image as Image

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

label_dict = {113: "糙纬", 75: "停车痕", 38: "油污"}


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
            im = np.array(im).astype("int32")
            im_label = Image.open(name_path[img_id]).convert("L")
            im_label = np.array(im_label).astype("int32")
            im_mask = np.where((im_label.flatten()) > 0)[0]
            im_mask = np.array(im_mask).astype("int32")
            yield im, im_label, im_mask

    return yield_one_data


# 初始化执行器以及程序
place = fluid.CUDAPlace(0) if conf["use_cuda"] else fluid.CPUPlace()
exe = fluid.Executor(place)
start_prog = fluid.Program()
train_prog = fluid.Program()

with fluid.program_guard(train_prog, start_prog):
    ipt_img = fluid.data(name="ipt_img", shape=conf["ipt_img_shape"], dtype="float32")

    ipt_label1 = fluid.data(name="ipt_label1", shape=conf["ipt_label1_shape"], dtype="int32")
    ipt_label2 = fluid.data(name="ipt_label2", shape=conf["ipt_label2_shape"], dtype="int32")
    ipt_label3 = fluid.data(name="ipt_label3", shape=conf["ipt_label3_shape"], dtype="int32")
    ipt_mask1 = fluid.data(name="mask1", shape=[-1, 1], dtype="int32", lod_level=1)
    ipt_mask2 = fluid.data(name="mask2", shape=[-1, 1], dtype="int32", lod_level=1)
    ipt_mask3 = fluid.data(name="mask2", shape=[-1, 1], dtype="int32", lod_level=1)
    ipt_label_list = [ipt_label1, ipt_label2, ipt_label3]
    ipt_mask_list = [ipt_mask1, ipt_mask2, ipt_mask3]
    loss = build_net(ipt_img, 3, ipt_label_list, ipt_mask_list)
    learning_rate = fluid.layers.exponential_decay(learning_rate=conf["e_learning_rate"],
                                                   decay_steps=epoch_data_count // conf["batch_size"],
                                                   decay_rate=0.1,
                                                   staircase=True)
    opt = fluid.optimizer.Adam(learning_rate=learning_rate)
    opt.minimize(loss)

# 读取设置

train_reader = paddle.batch(reader=paddle.reader.shuffle(reader(), 500), batch_size=conf["batch_size"])
feeder = fluid.DataFeeder(place=place, feed_list=[ipt_img, ipt_boxs, ipt_label])

# 训练部分
exe.run(start_prog)
fluid.io.load_params(executor=exe, dirname="./model/train_od_217", main_program=train_prog)

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
    # for data_id, data in enumerate(eval_reader()):
    #     eval_out = exe.run(program=eval_prog,
    #                        feed=feeder.feed(data),
    #                        fetch_list=[cur_map, accum_map])
    #
    # map_eval.reset(exe)
    # print("Epoch:", epoch + 1, "loss:", train_out[0], "cur_map:", eval_out[0], "accum_map:", eval_out[1])
