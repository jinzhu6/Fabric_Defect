# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import time
import paddle.fluid as fluid
import paddle
import numpy as np
import tools.osTool as osTool

from tools import imgTool as imgTool
from tools import labelTool as labelTool
from build_net import build_net

# 读取配置文件
with open("./config/train_full.yaml", "r", encoding="utf-8") as f:
    conf = f.read()
    conf = dict(yaml.load(conf, Loader=yaml.FullLoader))

dir_path = conf["data_path"] + "/"
train_dir_path = dir_path + "train"
eval_dir_path = dir_path + "eval"

# 数据读取
train_label_dict = labelTool.read_label(train_dir_path)
eval_label_dict = labelTool.read_label(eval_dir_path)


def reader(mode: str = "Train"):
    """
    数据读取器
    :param mode: 读取模式 Train或 Eval
    :return: reader
    """

    def yield_one_data():
        if mode == "Train":
            path = train_dir_path
            label_dict = train_label_dict
            for_test = False
        else:
            path = eval_dir_path
            label_dict = eval_label_dict
            for_test = True
        img_pretreatment_tool = imgTool.ImgPretreatment(path, for_test=for_test)
        for index in range(len(img_pretreatment_tool)):
            now_img_name = img_pretreatment_tool.img_files_name[index]
            img_pretreatment_tool.img_init(index, label_location_info=label_dict[now_img_name])
            img_list, label_list = img_pretreatment_tool.req_result()
            for im, label_info in zip(img_list, label_list):
                w, h = im.size
                im = np.array(im).transpose((2, 0, 1)).reshape(1, 1, h, w) * 0.007843
                label = np.array(label_info[0])
                box = np.array(label_info[1])
                yield im, box, label

    return yield_one_data


# 初始化执行器以及程序
place = fluid.CUDAPlace(0) if conf["use_cuda"] else fluid.CPUPlace()
exe = fluid.Executor(place)
start_prog = fluid.Program()
train_prog = fluid.Program()

with fluid.program_guard(train_prog, start_prog):
    ipt_img = fluid.layers.data(name="ipt_img", shape=[1] + conf["ipt_img_size"], dtype="float32")
    ipt_box_list = fluid.layers.data(name="ipt_box_list", shape=[4], dtype="float32", lod_level=1)
    ipt_label = fluid.layers.data(name="ipt_label", shape=[1], dtype="int32", lod_level=1)
    eval_prog = train_prog.clone(for_test=True)
    loss = build_net(ipt_img, ipt_box_list, ipt_label, mode="Train")
    opt = fluid.optimizer.Adam(learning_rate=conf["e_learning_rate"])
    opt.minimize(loss)

with fluid.program_guard(eval_prog, start_prog):
    map_eval = build_net(ipt_img, ipt_box_list, ipt_label, mode="Eval")
    cur_map, accum_map = map_eval.get_map_var()

# 读取设置

train_reader = paddle.batch(reader=paddle.reader.shuffle(reader(), 500), batch_size=conf["batch_size"])
eval_reader = paddle.batch(reader=paddle.reader.shuffle(reader(mode="Eval"), 500), batch_size=conf["batch_size"])
feeder = fluid.DataFeeder(place=place, feed_list=[ipt_img, ipt_box_list, ipt_label])

# 训练部分
exe.run(start_prog)

for epoch in range(conf["epochs"]):
    train_out = []
    eval_out = []
    start_time = time.time()
    step = 0
    for data_id, data in enumerate(train_reader):
        train_out = exe.run(program=train_prog,
                            feed=feeder.feed(data),
                            fetch_list=[loss])
        step = data_id
    cost_time = time.time() - start_time
    for data_id, data in enumerate(eval_reader):
        eval_out = exe.run(program=train_prog,
                           feed=feeder.feed(data),
                           fetch_list=[cur_map, accum_map])

        map_eval.reset(exe)
    print("Epoch:", epoch, "loss:", train_out[0], "cur_map:", eval_out[0], "accum_map:", eval_out[1])
    print("one data training time is:", cost_time / (step + 1) / conf["batch_size"])
