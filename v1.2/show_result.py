# 绘制结果图
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

# 将连续的滑动窗口内数据堆叠起来以作为 median_and_percentile 函数的输入,
# 从而求出窗口内数据的均值与波动范围
# |<- window->|
# |--- ... ---|         第一组
#   |--- ... ---|       第二组
#     |--- ... ---|     第三组
# ...                      ⋮
#
# 将这些组重新对齐, 构成的二维矩阵就可以作为 median_and_percentile 的输入了,
# 相应的输出就是 对应的 均值、10%~90% 波动范围的上下界，然后数据就可以用来画区域图


def median_and_percentile(x, axis, lower=20, upper=80):
    """
    计算中位数和指定的百分位数，目前设定：下界为20%，上界为80%
    :param x: 输入数据
    :param axis: 需要计算的轴
    :param lower: 指定百分位数的下界
    :param upper: 指定百分位数的上界
    :return: 中位数、平均数、下界值、上界值
    """
    assert (lower >= 0 and upper <= 100)
    median = np.median(x, axis)
    mean = np.mean(x, axis)
    low_per = np.percentile(x, lower, axis)
    up_per = np.percentile(x, upper, axis)
    return median, mean, low_per, up_per

def stack_data(x, window, stride):
    """
    按上述的算法将一维数组按窗口大小滑动堆叠构成二维数组
    :param x: 原始数据, 一维数组
    :param window: 窗口大小
    :param stride: 步长
    :return: 二维数组
    """
    n = len(x)
    assert n >= window
    y = []
    for i in range((n - window + 1) // stride):
        y.append(x[i * stride: i * stride + window])
    return np.asarray(y)


# 载入数据
with open('./checkpoints/x_reward.pkl', 'rb') as f:
    x_axis1 = pickle.load(f)
with open('./checkpoints/y_reward.pkl', 'rb') as f:
    y_axis1 = pickle.load(f)

# with open('./results/x_axis2.pkl', 'rb') as f:
#     x_axis2 = pickle.load(f)
# with open('./results/y_axis2.pkl', 'rb') as f:
#     y_axis2 = pickle.load(f)
#
# with open('./results/x_axis3.pkl', 'rb') as f:
#     x_axis3 = pickle.load(f)
# with open('./results/y_axis3.pkl', 'rb') as f:
#     y_axis3 = pickle.load(f)
#
# with open('./results/x_axis4.pkl', 'rb') as f:
#     x_axis4 = pickle.load(f)
# with open('./results/y_axis4.pkl', 'rb') as f:
#     y_axis4 = pickle.load(f)


# 让曲线呈现下降的趋势
# for i in range(len(y_axis1)):
#     y_axis1[i] *= -1
# for i in range(len(y_axis3)):
#     y_axis3[i] *= -1

# ------------时间复杂度比较图------------
# fig = plt.figure()
# spl = fig.add_subplot(111)
#
# # 相关数据经实验测量得到
# DRL = np.array([12.23, 12.97, 13.56, 14.69, 15.33, 16.29, 17.27, 18.40])
# MN_1 = np.array([50, 100, 150, 200, 250, 300, 350, 400])
# z_1 = np.polyfit(MN_1, DRL, 2)
# p_1 = np.poly1d(z_1)
# y_pred_1 = p_1(MN_1)
#
# p1 = spl.scatter(MN_1, DRL, marker='o', color='C3', label='Optimization-driven DDPG')
# spl.plot(MN_1, y_pred_1, color='C3', linestyle='--', label="_no_legend_")
#
# # 相关数据经实验测量得到
# SDR = np.array([9.775, 63, 57.69, 204, 254, 472, 751, 1229])
# MN_2 = np.array([50, 100, 150, 200, 250, 300, 350, 400])
# z_2 = np.polyfit(MN_2, SDR, 3)
# p_2 = np.poly1d(z_2)
# y_pred_2 = p_2(MN_2)
#
# p2 = spl.scatter(MN_2, SDR, marker='*', color='b', s=80, label='SDR-based Optimization')
# spl.plot(MN_2, y_pred_2, color='b', linestyle='--', label="_no_legend_")
#
# plt.xlabel(r'$M \times N$', fontsize=14)
# plt.ylabel("Run time (milliseconds)", fontsize=14)
# plt.legend([p1, p2], ["Optimization-driven DDPG", "SDR-based Optimization"], loc="upper left", frameon=False, fontsize=14)
# fig.tight_layout()
# pp = PdfPages('./result_figs/Run time.pdf')
# plt.savefig(pp, format='pdf')
# pp.close()
# plt.show()

# ------------迭代收敛比较图------------
y_axis1_ = stack_data(y_axis1, window=60, stride=3)
optm_m_1, optm_mean_1, optm_l_1, optm_u_1 = median_and_percentile(y_axis1_, axis = 1)
optm_x_1 = np.asarray(range(len(optm_m_1)))
x_ep1 = optm_x_1 * 200000 / len(optm_x_1)

# y_axis3_ = stack_data(y_axis3, window=60, stride=3)
# free_m_3, free_mean_3, free_l_3, free_u_3 = median_and_percentile(y_axis3_, axis = 1)
# free_x_3 = np.asarray(range(len(free_m_3)))
# x_ep3 = free_x_3 * 200000 / len(free_x_3)

fig = plt.figure()
spl = fig.add_subplot(111)

log_1 = []
log_2 = []
log_1_u = []
log_1_l = []
log_2_u = []
log_2_l = []
temp = 0

for i in range(len(optm_mean_1)):
    temp = 10 * math.log10(optm_mean_1[i])
    log_1.append(temp)
for i in range(len(optm_mean_1)):
    temp = 10 * math.log10(optm_u_1[i])
    log_1_u.append(temp)
for i in range(len(optm_mean_1)):
    temp = 10 * math.log10(optm_l_1[i])
    log_1_l.append(temp)
# for i in range(len(free_mean_3)):
#     temp = 10 * math.log10(free_mean_3[i])
#     log_2.append(temp)
# for i in range(len(free_mean_3)):
#     temp = 10 * math.log10(free_u_3[i])
#     log_2_u.append(temp)
# for i in range(len(free_mean_3)):
#     temp = 10 * math.log10(free_l_3[i])
#     log_2_l.append(temp)

spl.plot(x_ep1, log_1, color='C3', label='single-user-driven DDPG')
spl.fill_between(x_ep1, log_1_u, log_1_l, facecolor='C3', alpha=0.3)

# spl.plot(x_ep3, log_2, color='b', label="Model-free DDPG")
# spl.fill_between(x_ep3, log_2_u, log_2_l, facecolor='b', alpha=0.3)

plt.xlabel('Episode', fontsize=14)
plt.ylabel("reward", fontsize=14)
spl.legend(loc="lower right", frameon=False, fontsize=14)

fig.tight_layout()

pp = PdfPages('checkpoints/Convergence01.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.show()

# ------------方差比较图------------
y_axis1_ = stack_data(y_axis1, window = 60, stride = 3)
optm_m_1, optm_mean_1, optm_l_1, optm_u_1 = median_and_percentile(y_axis1_, axis = 1)
optm_x_1 = np.asarray(range(len(optm_m_1)))
x_ep1 = optm_x_1 * 200000 / len(optm_x_1)

# y_axis3_ = stack_data(y_axis3, window = 60, stride = 3)
# free_m_3, free_mean_3, free_l_3, free_u_3 = median_and_percentile(y_axis3_, axis = 1)
# free_x_3 = np.asarray(range(len(free_m_3)))
# x_ep3 = free_x_3 * 200000 / len(free_x_3)

fig = plt.figure()
spl = fig.add_subplot(111)

a_1 = np.array(log_1)
b_1 = np.array(log_1_u)
c_1 = np.array(log_1_l)

# a_2 = np.array(log_2)
# b_2 = np.array(log_2_u)
# c_2 = np.array(log_2_l)

variance_1 = (b_1 - a_1) * (b_1 - a_1) + (a_1 - c_1) * (a_1 - c_1)
#variance_2 = (b_2 - a_2) * (b_2 - a_2) + (a_2 - c_2) * (a_2 - c_2)

spl.plot(x_ep1, variance_1, color='C3', label='single-user-driven DDPG')
#spl.plot(x_ep3, variance_2, color='b', label='Model-free DDPG')

plt.xlabel('Episode', fontsize=14)
plt.ylabel("Variance", fontsize=14)
spl.legend(loc="upper right", frameon=False, fontsize=14)

fig.tight_layout()

pp = PdfPages('./checkpoints/Variance.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.show()

# ------------反射系数比较图------------
# y_axis2_ = stack_data(y_axis2, window = 1500, stride = 4)
# optm_m_2, optm_mean_2, optm_l_2, optm_u_2 = median_and_percentile(y_axis2_, axis = 1)
# optm_x_2 = np.asarray(range(len(optm_m_2)))
# x_ep2 = optm_x_2 * 200000 / len(optm_x_2)
# #
# y_axis4_ = stack_data(y_axis4, window = 1500, stride = 4)
# free_m_4, free_mean_4, free_l_4, free_u_4 = median_and_percentile(y_axis4_, axis = 1)
# free_x_4 = np.asarray(range(len(free_m_4)))
# x_ep4 = free_x_4 * 200000 / len(free_x_4)
#
# fig = plt.figure()
# spl = fig.add_subplot(111)
#
# spl.plot(x_ep2, optm_mean_2, color='C3', label='Optimization-driven DDPG')
# spl.plot(x_ep4, free_mean_4, color='b', label='Model-free DDPG')
#
# temp1 = []
# temp2 = []
# for i in range(len(optm_mean_2)):
#     temp1.append(optm_mean_2[i][0])
#
# for i in range(len(free_mean_4)):
#     temp2.append(free_mean_4[i][0])

# plt.xlabel('Episode', fontsize=14)
# plt.ylabel(r' $\rho$ '"-- the magnitude of reflection", fontsize=14)
# spl.legend(loc="upper right", frameon=False, fontsize=14)

# fig.tight_layout()
#
# pp = PdfPages('./result_figs/rho.pdf')
# plt.savefig(pp, format='pdf')
# pp.close()
# plt.show()
