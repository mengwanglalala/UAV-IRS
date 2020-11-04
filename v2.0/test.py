import matplotlib.pyplot as plt
import numpy as np
n = 5

x = [0 , 10, 5, 10, 6]
y = [0 , -5, -5, 0, 6]

x=np.array(x)

y=np.array(y)

fig = plt.figure(1)

#colors = np.random.rand(n) # 随机产生10个0~1之间的颜色值，或者
colors = ['r', 'g', 'g', 'g', 'b']  # 可设置随机数取

area = [300,100,100,100,1]

widths = np.arange(n) #0-9的数字

plt.scatter(x, y, s=area, c=colors, linewidths=widths, alpha=0.5, marker='o')

# 设置X轴标签
plt.xlabel('X坐标')
# 设置Y轴标签
plt.ylabel('Y坐标')
plt.title('test绘图函数')

# 设置横轴的上下限值
plt.xlim(-5, 15)
# 设置纵轴的上下限值
plt.ylim(-10, 10)

# 设置横轴精准刻度
plt.xticks(np.arange(np.min(x) - 0.2, np.max(x) + 0.2, step=0.3))
# 设置纵轴精准刻度
plt.yticks(np.arange(np.min(y) - 0.2, np.max(y) + 0.2, step=0.3))

# 设置横轴精准刻度
plt.xticks(np.arange(-5, 15, step=1))
# 设置纵轴精准刻度
plt.yticks(np.arange(-10, 10, step=1))

#plt.annotate("(" + str(round(x[2],2)) +", "+ str(round(y[2],2)) +")", xy=(x[2], y[2]), fontsize=10, xycoords='data')  #或者
plt.annotate("({0},{1})".format(round(x[2],2), round(y[2],2)), xy=(x[2], y[2]), fontsize=10, xycoords='data')
# xycoords='data' 以data值为基准
# 设置字体大小为 10
#plt.text(round(x[4],2), round(y[4],2), "UAV", fontdict={'size': 10, 'color': 'blue'})  # fontdict设置文本字体
# Add text to the axes.
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.legend(['UAV轨迹'], loc=2, fontsize = 10)
#plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
plt.show()