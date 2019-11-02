# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import paddle.fluid as fluid

from build_net import build_net

# 读取配置文件
with open("./config/train_full.yaml", "r", encoding="utf-8") as f:
    conf = f.read()
    conf = yaml.load(conf, Loader=yaml.FullLoader)


# 数据读取
def reader(mode: str = "Train"):
    """
    数据读取器
    :param mode: 读取模式 Train或 Eval
    :return: reader
    """

    def yield_one_data():
        pass

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


# 训练部分
exe.run(start_prog)

for epoch in range(conf["epochs"]):
    train_out = exe.run(program=train_prog,
                        feed={},
                        fetch_list=[loss])
    eval_out = exe.run(program=train_prog,
                       feed={},
                       fetch_list=[cur_map, accum_map])
map_eval.reset(exe)
