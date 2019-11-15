# Author:  Acer Zhang
# Datetime:2019/11/3
# Copyright belongs to the author.
# Please indicate the source for reprinting.

def replace_label(array_obj):
    """
    替换Label中颜色
    :param array_obj: Numpy的array对象
    :return: 替换颜色后的Numpy的array对象
    """
    # color_dict = {38:"正常", 14: "糙纬", 113: "停车痕", 75: "油污", 52: Other}
    color_dict = {38: 1, 14: 1, 113: 2, 75: 3, 52: 4}
    for color_key, color_value in color_dict.items():
        array_obj[array_obj == color_key] = color_value
    return array_obj
