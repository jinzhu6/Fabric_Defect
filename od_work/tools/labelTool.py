# Author:  Acer Zhang
# Datetime:2019/11/3
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import json
from tools import osTool as osTool

label_key = {"油污/浆斑-类": 1,
             "停车痕-类": 2,
             "糙纬-类": 3,
             "其它瑕疵-类": 4}


def load_labelme_json(json_file_path: str):
    """
    从lableme中读取数据
    :param json_file_path:Json所在的目录
    :return: [[label,box],...]
    """
    with open(json_file_path, "r", encoding="GBK") as f:
        infos = json.loads(f.read())
    infos = infos["shapes"]
    tmp = []
    for info in infos:
        label = label_key[info["label"]]
        box = info["points"][0] + info["points"][1]
        tmp.append([label, box])
    return tmp


def read_label(path):
    """
    读取标签信息
    :param path: 带有labelme标注的json文件的目录
    :return: 标签字典 {文件名:[[label,box],...]}
    """
    json_names, json_file_paths = osTool.read_ext_in_dir(path, ext="json", name_none_ext=True)
    tmp = {}
    for json_name, json_file_path in zip(json_names, json_file_paths):
        info = load_labelme_json(json_file_path)
        tmp[json_name] = info
    return tmp


