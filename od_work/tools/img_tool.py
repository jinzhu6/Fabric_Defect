# Author:  Acer Zhang
# Datetime:2019/10/26
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os
import sys
import random
import traceback

from PIL import Image, ImageEnhance
import numpy as np

from tools.os_tool import read_ext_in_dir

fontP = "./font/1.ttf"


class ImgPretreatment:
    """

    图像预处理类

    批量处理代码1 分类任务:
        # 创建工具对象并传入相关参数
        img_pretreatment_tool=ImgPretreatment(args)
        # 处理所有图像
        for index in range(img_pretreatment_tool.__len__())

            # 初始化当前index图像
            img_pretreatment_tool.img_init(index)
            # 对图像进行操作
            img_pretreatment_tool.img_cut_color()
            img_pretreatment_tool.img_xxx()
            img_pretreatment_tool.img_xxx()
            ...
            # 获取最终处理好的数据(Pillow对象)
            final_img_pil_obj,final_loc_box = img_pretreatment_tool.req_result()
    批量处理代码2 目标检测任务:
        # 创建工具对象并传入相关参数
        img_pretreatment_tool=ImgPretreatment(args)
        # 处理所有图像
        for index in range(img_pretreatment_tool.__len__())
            # 可以获取当前初始化的文件名(不含扩展名) 防止与将要操作的图片不一致
            now_img_name = self.img_files_name[index]
            loc = xxx
            # 初始化当前index图像
            img_pretreatment_tool.img_init(index，loc)
            # 对图像进行操作
            img_pretreatment_tool.img_cut_color()
            img_pretreatment_tool.img_xxx()
            img_pretreatment_tool.img_xxx()
            ...
            # 获取最终处理好的数据(Pillow对象)
            final_img_pil_obj,final_loc_box = img_pretreatment_tool.req_result()

    批量处理代码3 从内存读取:
        # 只需要在初始化图片时传入PIL对象即可 但部分功能可能无法使用(颜色裁剪，若没有之前保留的均值参数则无法计算颜色均值)
        # 创建工具对象并传入相关参数
        img_pretreatment_tool=ImgPretreatment(args)
        # 处理所有图像
        for index in range(len(img_pretreatment_tool))
            # 初始化当前index图像
            img_pretreatment_tool.img_init(index，pil_obj)
            # 对图像进行操作
            img_pretreatment_tool.img_cut_color()
            img_pretreatment_tool.img_xxx()
            img_pretreatment_tool.img_xxx()
            ...
            # 获取最终处理好的数据(Pillow对象)
            final_img_pil_obj,final_loc_box = img_pretreatment_tool.req_result()
    Tips:
    1、第一次进行图像减颜色均值操作时过程较长，并非卡顿。
    2、如果要统一数据集尺寸，则最好放在所有对图像的操作之前，初始化图像操作之后。这样更有利运行速度。
    3、若进行颜色裁剪，则不可以进行保存图片操作，请在该操作前进行保存。
    4、在传入标签位置信息时不用担心与当前所初始化的图片是否对应，因为在初始化该类时就已经生成了图片路径索引表
        只需要获取 self.img_files_name 就可以知道当前index对应的图片名。
    参数保存:
    在获取颜色均值后会保一个参数文件'img_pretreatment.txt'记录参数。
    参数读取仅在for_test=True或ignore_log=False时生效。

    设计思想:
    ->给定文件夹以及参数进行读取。
    ->构建图片列表。
    ->给出索引值，初始化对应图片，若包含位置信息则在此传入。
    ->对图片进行操作。
    ->获取当前处理后的图片。

    单张变多张操作: 初始化图片后，内部会以一个列表存储该图片，若涉及单张变多张的操作则会把结果都保存在该列表中。
    尺寸变换操作: 每次处理都会有一个全局变量来记录变换后的尺寸
    颜色裁剪操作: 仅计算文件夹内图片的颜色均值，无论怎样调整操作顺序，该操作总是最后一个进行，而且较为独立。
    """

    def __init__(self, all_img_path, mean_color_num=500, dir_deep=0, img_type="jpg", read_img_type='L',
                 ignore_log=False, for_test=False, debug=True):
        """
        :param all_img_path: 图像文件所在文件夹路径
        :param mean_color_num: 颜色均值采样数
        :param dir_deep: 检索文件夹的深度
        :param img_type: 图像文件类型，默认jpg格式
        :param read_img_type: 图像读取通道类型 默认为灰度
        :param ignore_log: 是否忽略之前参数文件
        :param for_test: 是否用于测试，如果用于测试则自动读取颜色裁剪信息
        :param debug: 设置为False后将进入哑巴模式，什么信息都不会打印在屏幕上，报错信息除外
        """

        if debug:
            print("----------ImgPretreatment Start!----------")

        self.img_files_name, self.img_files_path = read_ext_in_dir(all_img_path, dir_deep, img_type, name_none_ext=True)

        self.len_img = len(self.img_files_path)
        assert self.len_img != 0, "No file is in the folder."
        self.mean_color_num = mean_color_num
        self.img_type = img_type
        self.read_img_type = read_img_type
        self.debug = debug
        self.shape = Image.open(self.img_files_path[0]).size

        # Flag 变量
        self.allow_req_img = False  # 图像初始化控制
        self.allow_save_flag = True  # 允许保存，如果进行去均值操作则不允许保存
        self._color_mean_flag = False  # 是否需要计算颜色均值
        self.__need_color_cut_flag = False  # 是否需要颜色去均值
        self.__first_print_flag = True  # 是否第一次输出控制变量
        self.__contain_location = False  # 是否包含位置标签信息
        if ignore_log is False or for_test is True:
            try:
                with open("./img_pretreatment.txt", "r") as f:
                    info = f.read().split("-")
                    check_info = str(self.read_img_type) + str(self.len_img) + str(self.mean_color_num)
                    if info[0] == check_info or for_test:
                        self._color_mean_flag = True
                        self.allow_save_flag = False
                        self.__color_mean = float(info[1][1:-1])
                        if debug:
                            print("Load Log Successfully!")
            except:
                assert not for_test, "Load Log Finally,Place check img_pretreatment.txt!"
        # 当前进程变量
        self.now_index = -1
        self.now_img_name = None
        self.now_img_file_path = None
        self.now_img_obj_list = []
        self.now_label_locs_list = []
        self.tmp_one_img_to_num = 0
        self.tmp_all_img_to_num = 0
        if debug:
            print("Data read successfully number:", self.len_img)

    def img_init(self, index, label_location_info=None, pil_obj=None):
        """
        图像初始化
        :param index: 需要初始化图像的索引
        :param label_location_info: 图像的位置信息 示例[[x_up, y_up, x_down, y_down],...]
        :param pil_obj: 若不想从文件中读取，请从此传入PIL对象
        """
        self.allow_save_flag = True
        self.__contain_location = False
        if pil_obj is not None:
            self._color_mean_flag = False
            self.now_img_obj_list = []
            self.now_img_obj_list.append(pil_obj)
        if index is not self.now_index or len(self.now_img_obj_list) != 1:
            try:
                self.now_img_obj_list = []
                self.now_img_obj_list.append(Image.open(self.img_files_path[index]).convert(self.read_img_type))
                self.now_index = index
                self.now_img_name = self.img_files_name[self.now_index]
            except:
                print(traceback.format_exc())
        if label_location_info is not None:
            self.__contain_location = True
            self.now_label_locs_list = []
            self.now_label_locs_list.append(label_location_info)

    def __color_mean_start(self):
        """
        颜色均值获取
        :return:颜色均值
        """
        sum_img_numpy = np.zeros((1, 1, 1), dtype=np.float)
        if self.read_img_type is "L":
            sum_img_numpy = np.zeros(1, dtype=np.float)

        self.__color_mean = [0, 0, 0]
        mean_color_num = self.mean_color_num
        success_num = 1
        only_shape = None
        for id_, imgP in enumerate(self.img_files_path):
            im = Image.open(imgP).convert(self.read_img_type)
            if id_ == 0:
                only_shape = im.size
            if only_shape != im.size:
                mean_color_num += 1
                continue
            sum_img_numpy += np.mean(np.asarray(im).reshape((1, im.size[0], im.size[1])))
            success_num += 1
            if id_ == mean_color_num or id_ == self.len_img - 1:
                self.__color_mean = np.around((sum_img_numpy / success_num), decimals=3).tolist()
                self._color_mean_flag = True

                with open("./img_pretreatment.txt", "w") as f:
                    f.write(
                        (str(self.read_img_type) + str(self.len_img) + str(self.mean_color_num) + "-" + str(
                            self.__color_mean)))
                    if self.debug is True:
                        print("Color mean :", self.__color_mean)
                    print("Successful counting in color mean:", success_num - 1)
                    print("Write log --Done!")
                return self.__color_mean

    def img_cut_color(self):
        """
        颜色去均值操作
        防止因为去均值后颜色模式发生改变，所以立了个Flag，使它即将结束时运行该操作。
        """
        self.__need_color_cut_flag = True
        self.allow_save_flag = False

    def img_only_one_shape(self, expect_w, expect_h):
        """
        传入图片将中心修正为数据集统一的尺寸
        注意！期望大小必须小于原始图片大小
        """
        temp_img_list = []
        for index, now_img_obj in enumerate(self.now_img_obj_list):
            w, h = now_img_obj.size
            assert (w - expect_w >= 0 or h - expect_h >= 0), (
                "Expected values are larger than the original image size. Please adjust the expected values to be "
                "smaller than the original size!")
            box = ((w - expect_w) // 2, (h - expect_h) // 2,
                   (w - expect_w) // 2 + expect_w, (h - expect_h) // 2 + expect_h)
            img = now_img_obj.crop(box)
            temp_img_list.append(img)
            if self.__contain_location:
                self.__repair_loc(index, box)

        self.now_img_obj_list = temp_img_list
        self.shape = temp_img_list[0].size

    def img_resize(self, expect_w, expect_h):
        """
        多相位调整图像大小
        :param expect_w: 期望的宽度-横向
        :param expect_h: 期望的高度-纵向
        """
        temp_list = []
        for index, now_img_obj in enumerate(self.now_img_obj_list):
            if self.__contain_location:
                w, h = now_img_obj.size
                w_, h_ = (expect_w / w, expect_h / h)
                for loc_id, loc in enumerate(self.now_label_locs_list[index]):
                    tmp_loc = [loc[0], [loc[1][0] * w_, loc[1][1] * w_, loc[1][2] * h_, loc[1][3] * w_]]
                    self.now_label_locs_list[index][loc_id] = tmp_loc
            img = now_img_obj.resize((expect_w, expect_h), Image.LANCZOS)
            temp_list.append(img)

        self.now_img_obj_list = temp_list
        self.shape = temp_list[0].size

    def img_rotate(self, angle_range=(0, 0), angle_step=1, angle_and_transpose=False, only_transpose=True):
        """
        图像翻转
        如果仅返回规则翻转，则不需要修改前两个参数
        :param angle_range:旋转最小角度和最大角度
        :param angle_step:选择角度之间的步长
        :param angle_and_transpose:旋转角度之后再进行水平和垂直旋转
        :param only_transpose:仅进行水平和垂直旋转
        Tips:如果使用该模块，则只在最后获取时运行
        """

        def tran(ori_pil_obj, out_pil_list):
            out_pil_list.append(ori_pil_obj)
            out_pil_list.append(ori_pil_obj.transpose(Image.FLIP_LEFT_RIGHT))
            out_pil_list.append(ori_pil_obj.transpose(Image.FLIP_TOP_BOTTOM))

        temp_list = []
        if only_transpose is True:
            tmp_locs = []
            for index, now_img_obj in enumerate(self.now_img_obj_list):
                tran(now_img_obj, temp_list)
                w, h = now_img_obj.size
                tmp_loc1 = []
                for loc in self.now_label_locs_list[index]:
                    tmp_loc1.append(loc)
                tmp_loc2 = []
                for loc in self.now_label_locs_list[index]:
                    tmp_loc2.append([loc[0], [w - loc[1][0], loc[1][1], w - loc[1][2], loc[1][3]]])
                tmp_loc3 = []
                for loc in self.now_label_locs_list[index]:
                    tmp_loc3.append([loc[0], [loc[1][0], h - loc[1][1], loc[1][2], h - loc[1][3]]])
                tmp_locs.append(tmp_loc1)
                tmp_locs.append(tmp_loc2)
                tmp_locs.append(tmp_loc3)
            self.now_label_locs_list = tmp_locs

        else:
            print("暂时不需要该功能")
            """
            for now_img_obj in self.now_img_obj_list:
                for angle in range(angle_range[0], angle_range[1], angle_step):
                    temp_list.append(now_img_obj.rotate(angle))
            if angle_and_transpose is True:
                self.now_img_obj_list = []
                for now_img_obj in temp_list:
                    tran(now_img_obj, self.now_img_obj_list)

            """
        self.now_img_obj_list = temp_list

    def img_random_crop(self, expect_w: int, expect_h: int, random_num: int = 1):
        """
        随机裁剪
        期望值需小于当前图片尺寸
        :param expect_w:
        :param expect_h:
        :param random_num:
        """
        temp_img_list = []
        for seed in range(1, random_num + 1):
            for index, now_img in enumerate(self.now_img_obj_list):
                w, h = now_img.size
                seed_w = random.randint(0, (w - expect_w))
                seed_h = random.randint(0, (h - expect_h))
                box = (seed_w, seed_h, seed_w + expect_w, seed_h + expect_h)
                if self.__contain_location:
                    self.__repair_loc(index, box)
                temp_img_list.append(now_img.crop(box))
        self.now_img_obj_list = temp_img_list
        self.shape = temp_img_list[0].size

    def img_random_contrast(self, random_num: int = 1, lower=0.5, upper=1.5):
        """
        随机对比度
        :param random_num: 随机次数，尽可能在3以内，建议为1，均匀随机
        :param lower:最低可能的对比度
        :param upper:最高可能的对比度
        """

        temp_list = list(self.now_img_obj_list)
        temp_los = list(self.now_label_locs_list) if self.__contain_location else None
        for seed in range(1, random_num + 1):
            factor = random.uniform(lower + ((upper - lower) * seed - 1 / random_num),
                                    lower + ((upper - lower) * seed / random_num))
            for index, now_img_obj in enumerate(self.now_img_obj_list):
                img = ImageEnhance.Sharpness(now_img_obj)
                img = img.enhance(factor)
                temp_list.append(img)
                if self.__contain_location:
                    temp_los.append(temp_los[index])
        self.now_img_obj_list = temp_list
        self.now_label_locs_list = temp_los

    def img_random_brightness(self, random_num: int = 1, lower=0.5, upper=1.5):
        """
        随机亮度
        :param random_num: 随机次数，尽可能在3以内，建议为1，均匀随机
        :param lower:最低可能的亮度
        :param upper:最高可能的亮度
        """

        temp_list = list(self.now_img_obj_list)
        temp_los = list(self.now_label_locs_list) if self.__contain_location else None
        for seed in range(1, random_num + 1):
            factor = random.uniform(lower + ((upper - lower) * seed - 1 / random_num),
                                    lower + ((upper - lower) * seed / random_num))
            for index, now_img_obj in enumerate(self.now_img_obj_list):
                img = ImageEnhance.Brightness(now_img_obj)
                img = img.enhance(factor)
                temp_list.append(img)
                if self.__contain_location:
                    temp_los.append(temp_los[index])
        self.now_img_obj_list = temp_list
        self.now_label_locs_list = temp_los

    def img_random_saturation(self, random_num: int = 1, lower=0.5, upper=1.5):
        """
        随机饱和度
        :param random_num: 随机次数，尽可能在3以内，建议为1，均匀随机
        :param lower:最低可能的亮度
        :param upper:最高可能的亮度
        """
        temp_list = list(self.now_img_obj_list)
        temp_los = list(self.now_label_locs_list) if self.__contain_location else None
        for seed in range(1, random_num + 1):
            factor = random.uniform(lower + ((upper - lower) * seed - 1 / random_num),
                                    lower + ((upper - lower) * seed / random_num))
            for index, now_img_obj in enumerate(self.now_img_obj_list):
                img = ImageEnhance.Color(now_img_obj)
                img = img.enhance(factor)
                temp_list.append(img)
                if self.__contain_location:
                    temp_los.append(temp_los[index])
        self.now_img_obj_list = temp_list
        self.now_label_locs_list = temp_los

    def req_result(self, save_path=None):
        """
        获取当前处理进程中图片
        :param save_path:如果保存图片，则需要提供保存路径
        :return: PIL_Obj_List or [PIL_Obj_List,location_info_list]
        """
        # 特殊操作区域
        if self.__need_color_cut_flag is True:
            temp_list = []
            for now_img_obj in self.now_img_obj_list:
                now_img_obj = Image.fromarray(np.asarray(now_img_obj) - self._req_color_mean())
                temp_list.append(now_img_obj)
            self.now_img_obj_list = list(temp_list)
            self.__need_color_cut_flag = False

        # 输出区域

        if self.debug and self.__first_print_flag:
            tmp_shape = self.shape
            self.tmp_one_img_to_num = len(self.now_img_obj_list)
            self.tmp_all_img_to_num = self.len_img * len(self.now_img_obj_list)
            print("The current size of the first image output is ", tmp_shape)
            print("The number of single image pre-processed is expected to be ", self.tmp_one_img_to_num)
            print("The total number of pictures expected to be produced is ", self.tmp_all_img_to_num)
            self.__first_print_flag = False
        if self.debug:
            self.__progress_print()
        # 保存区域
        if save_path is not None:
            assert self.allow_save_flag, "Can not save F mode img! Please undo img_cut_color operation!"
            folder = os.path.exists(save_path)
            if not folder:
                os.makedirs(save_path)
            if len(self.now_img_obj_list) != 1:
                for id_, img in enumerate(self.now_img_obj_list):
                    img.save(
                        os.path.join(save_path, self.img_files_name[self.now_index] + str(id_) + ".jpg").replace("\\",
                                                                                                                 "/"))
            else:
                self.now_img_obj_list[0].save(
                    os.path.join(save_path, self.img_files_name[self.now_index] + ".jpg").replace("\\", "/"))
        # 数据返回区域
        if self.__contain_location:
            return [self.now_img_obj_list, self.now_label_locs_list]
        else:
            return self.now_img_obj_list

    def req_img_count(self):
        """
        获取当前处理文件夹下处理后图像总数
        返回单张图片处理后的数量，文件夹下预计处理完后的数量
        :return: tmp_one_img_to_num, tmp_all_img_to_num
        """
        return self.tmp_one_img_to_num, self.tmp_all_img_to_num

    def _req_color_mean(self):
        if self._color_mean_flag:
            return self.__color_mean
        else:
            return self.__color_mean_start()

    def __repair_loc(self, index, box):
        """
        修正标签坐标，用于裁剪后坐标偏移修正。传入在原图基础上裁剪的范围框--box和当前坐标列表即可
        :param index: 当前处理列表的索引值
        :param box: 裁剪框
        """

        temp_loc_list = []
        for loc_id, loc in enumerate(self.now_label_locs_list[index]):
            final_loc = [loc[1][0] - box[0], loc[1][1] - box[1], loc[1][2] - box[0], loc[1][3] - box[1]]
            if min(final_loc) >= 0:
                temp_loc_list.append([loc[0], final_loc])
            else:
                temp_loc_list.append([0, [0, 0, 0, 0]])
        self.now_label_locs_list[index] = temp_loc_list

    def __progress_print(self):
        """
        打印进度百分比
        """

        percentage = (self.now_index + 1) / self.len_img
        stdout_obj = sys.stdout
        style = "|\\|"
        if self.now_index % 2 == 0:
            style = "|/|"
        if self.now_index == self.len_img - 1:
            stdout_obj.write('\rPercentage of progress:{:.2%}'.format(percentage))
            print("\n----------ImgPretreatment Done!-----------\n")
        else:
            stdout_obj.write('\r' + style + '  Percentage of progress:{:.2%}'.format(percentage))

    def __len__(self):
        return self.len_img

# # 测试代码
# all_img_tool = ImgPretreatment(all_img_path="./", debug=True, ignore_log=True)
# for i in range(all_img_tool.__len__()):
#     all_img_tool.img_init(i, [[5, 10, 15, 20]])
#     all_img_tool.img_rotate(only_transpose=True)
#     all_img_tool.img_random_brightness()
#     all_img_tool.img_random_contrast()
#     all_img_tool.img_cut_color()
#     all_img_tool.img_resize(450, 220)
#     all_img_tool.img_random_saturation()
#     all_img_tool.img_only_one_shape(448, 218)
#     all_img_tool.img_random_crop(440, 210, 2)
#     all_img_tool.req_result()
#     pass
