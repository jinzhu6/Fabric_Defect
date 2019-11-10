# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import time
import paddle.fluid as fluid
import paddle
import numpy as np

from tools import img_tool as imgTool
from tools import label_tool as labelTool
from build_net import build_net

# 读取配置文件
with open("./config/train_classsify.yaml", "r", encoding="utf-8") as f:
    conf = f.read()
    conf = dict(yaml.load(conf, Loader=yaml.FullLoader))

dir_path = conf["data_path"] + "/"
train_dir_path = dir_path + "train"
eval_dir_path = dir_path + "eval"
epoch_data_count = 50  # 默认每Epoch有多少图片数量
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
            img_pretreatment_tool.img_only_one_shape(300, 300)
            img_list, label_list = img_pretreatment_tool.req_result()
            global epoch_data_count
            if epoch_data_count == 50:
                epoch_data_count = img_pretreatment_tool.req_img_count()
            for im, label_infos in zip(img_list, label_list):
                w, h = im.size
                im = np.array(im).reshape(1, 1, h, w) * 0.007843
                label = np.array([i[0] for i in label_infos])
                box = np.array([[i[1][0] / w, i[1][1] / h, i[1][2] / w, i[1][3] / h] for i in label_infos])
                yield im, box, label

    return yield_one_data


# 初始化执行器以及程序
place = fluid.CUDAPlace(0) if conf["use_cuda"] else fluid.CPUPlace()
exe = fluid.Executor(place)
start_prog = fluid.Program()
train_prog = fluid.Program()

with fluid.program_guard(train_prog, start_prog):
    ipt_img = fluid.data(name="ipt_img", shape=[-1, 3] + conf["ipt_img_size"], dtype="float32")
    ipt_label = fluid.data(name="ipt_label", shape=[-1, 1], dtype="int32", lod_level=1)
    layers = build_net(ipt_img, mode="TandV2")
    eval_prog = train_prog.clone(for_test=True)
    loss = fluid.layers.cross_entropy(layers[-1], ipt_label)
    learning_rate = fluid.layers.exponential_decay(learning_rate=conf["e_learning_rate"],
                                                   decay_steps=epoch_data_count // conf["batch_size"],
                                                   decay_rate=0.5,
                                                   staircase=True)
    opt = fluid.optimizer.Adam(learning_rate=learning_rate)
    opt.minimize(loss, parameter_list=layers)

with fluid.program_guard(eval_prog, start_prog):
    acc = fluid.layers.accuracy(layers[-1], ipt_label)

# 读取设置

train_reader = paddle.batch(reader=paddle.reader.shuffle(reader(), 500), batch_size=conf["batch_size"])
eval_reader = paddle.batch(reader=paddle.reader.shuffle(reader(mode="TandV1"), 500), batch_size=conf["batch_size"])
feeder = fluid.DataFeeder(place=place, feed_list=[ipt_img, ipt_label])

# 训练部分
exe.run(start_prog)

print("start!")
for epoch in range(conf["epochs"]):
    train_out = []
    eval_out = []
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
    for data_id, data in enumerate(eval_reader()):
        eval_out = exe.run(program=eval_prog,
                           feed=feeder.feed(data),
                           fetch_list=[acc])

    print("Epoch:", epoch + 1, "loss:", train_out[0], "acc:", eval_out[0])
