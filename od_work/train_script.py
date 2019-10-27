# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import yaml
import paddle.fluid as fluid

import build_net

# 读取配置文件
with open("./config/net.yaml", "r", encoding="utf-8") as f:
    conf = f.read()
    conf = yaml.load(conf, Loader=yaml.FullLoader)

cur_map, accum_map = map_eval.get_map_var()
