# Author:  Acer Zhang
# Datetime:2019/11/12
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import time
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from PIL import ImageDraw
from tools import img_tool as img_tool

# Hyper parameter
use_cuda = False  # Whether to use GPU or not
batch_size = 100  # Number of incoming batches of data
model_path = "./model/infer_od_72"  # infer model path


def data_reader():
    def reader():
        img_pretreatment_tool = img_tool.ImgPretreatment("./data/test", ignore_log=True)
        for index in range(len(img_pretreatment_tool)):
            img_pretreatment_tool.img_init(index)
            img_pretreatment_tool.img_only_one_shape(2400, 1200)
            img_pretreatment_tool.img_resize(1200, 1200)
            im = img_pretreatment_tool.req_result()[0]
            w, h = im.size
            im = np.array(im).reshape(1, 1, h, w)
            yield im

    return reader


# Initialization

place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
startup = fluid.Program()

# load infer model
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)


def draw_bbox_image(img, nms_out):
    confs_threshold = 0.25  # 置信度
    draw = ImageDraw.Draw(img)
    for dt in nms_out:
        if dt[1] < confs_threshold:
            continue
        this_label = dt[0]
        bbox = dt[2:]
        # 根据网络输出，获取矩形框的左上角、右下角坐标相对位置
        draw.rectangle((bbox[0] * 2400, bbox[1] * 1200, bbox[2] * 2400, bbox[3] * 1200), None, 'red')
    img.show()


exe.run(startup)
# Start infer
infer_reader = paddle.batch(reader=data_reader(), batch_size=batch_size)
infer_feeder = fluid.DataFeeder(place=place, feed_list=feed_target_names, program=infer_program)

start_time = time.time()
for data in infer_reader():
    results = exe.run(infer_program, feed=infer_feeder.feed(data),
                      fetch_list=fetch_targets, return_numpy=False)
    nms_out = np.asarray(results[0])

    im = Image.open("./data/test/1.jpg")
    draw_bbox_image(im, nms_out)
    print(nms_out)
    pass
print(time.time() - start_time)  # [1]4.6 [10]4.5 [40]4.5 [100]41
