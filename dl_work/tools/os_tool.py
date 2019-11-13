# Author:  Acer Zhang
# Datetime:2019/11/3
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os


def read_ext_in_dir(path: str, dir_deep: int = 0, ext: str = 'jpg', name_none_ext: bool = False):
    """
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    :param dir_deep:文件夹检索深度，默认为0
    :param ext:指定文件类型，如果没有指定则视为jpg类型
    :param name_none_ext:如果为True则返回的文件名列表中不含有扩展名
    :return: nameL:文件夹内所有文件名+路径 '000030_1_0.jpg' or '000030_1_0' ,'./trainData/ori1/20181024/000030_1_0.jpg'
    """
    ext = "." + ext
    name_list = []  # 保存文件名
    name_path = []
    for id_, (root, dirs, files) in enumerate(os.walk(path)):
        for file in files:
            if os.path.splitext(file)[1] == ext:
                if name_none_ext is True:
                    name_list.append(os.path.splitext(file)[0])
                else:
                    name_list.append(file)
                name_path.append(str(os.path.join(root, file)).replace("\\", "/"))
        if id_ == dir_deep:
            break
    return name_list, name_path
