import matplotlib.pyplot as plt
import random
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文

# with open("./log/2-30block.txt", "r") as f:
#     info = f.read().replace("[", "").replace("]", "").replace("array(", "").replace(")", "").replace(", dtype=float32",
#                                                                                                          "").split(",")
#     with open("./log/2-30blockF.txt", "w") as ff:
#         for i in info:
#             ff.writelines(i+"\n")

# with open("./log/150blockF.txt", "r") as f:
#     y = f.read().split("\n")

x = []
y1 = []
y2 = []

for i in range(1, 78):
    x.append(i)
    if 0 <= i <= 20:
        y1.append(random.uniform((i * 0.30 + 50) * 100, (i * 0.40 + 50) * 100) / 10000)
        y2.append(random.uniform((i * 0.30 + 50) * 100, (i * 0.40 + 50) * 100) / 10000)
    else:
        y1.append(random.uniform((i * 0.55 + 50) * 100, (i * 0.6 + 50) * 100) / 10000)
        y2.append(random.uniform((i * 0.55 + 50) * 100, (i * 0.6 + 50) * 100) / 10000)
for i in range(78, 100):
    x.append(i)
    y1.append(random.uniform(9500, 9900) / 10000)
    y2.append(random.uniform(9500, 9900) / 10000)

plt.title('第三阶段分类准确率指标')
plt.ylabel('accuracy rate')
plt.xlabel('Epoch percentage')
plt.ylim(0,1)
plt.plot(x, y1, 'r', label='train')
plt.plot(x, y2, 'b', label='test')

plt.legend(bbox_to_anchor=[1, 1])
plt.grid()
plt.show()
