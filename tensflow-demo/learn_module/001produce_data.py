# 首先导入3个库
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 随机产生100个数据点,随机概率符合高斯分布(正态分布)
num_points = 100
vectors_set = []
for i in range(num_points):
    # Draw random samples from a normal (Gaussian) distribution.
    x1 = np.random.normal(0., 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    # 坐标点
    vectors_set.append([x1, y1])
# 定义特征向量x
x_data = [v[0] for v in vectors_set]
# 定义标签向量y
y_data = [v[1] for v in vectors_set]

# 按[x_data,y_data]在X-Y坐标系中以打点方式显示,调用plt建立坐标系并将值输出
# plt.scatter(x_data, y_data, c='b')
# plt.show()
tf.compat.v1.disable_v2_behavior()

# 利用TensorFlow随机产生w和b,为了图形显示需要,分别定义名称myw 和 myb
w = tf.Variable(tf.compat.v1.random_uniform([1], -1., 1.), name='myw')
b = tf.Variable(tf.zeros([1]), name='myb')
# 根据随机产生的w和b,结合上面随机产生的特征向量x_data,经过计算得出预估值
y = w * x_data + b
# 以预估值y和实际值y_data之间的均方差作为损失
loss = tf.reduce_mean(tf.square(y - y_data, name='mysquare'), name='myloss')
# 采用梯度下降法来优化参数
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name='mytrain')

# global_variables_initializer初始化Variable等变量
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print("w=", sess.run(w), "b= ", sess.run(b), sess.run(loss))
# 迭代20次train
for step in range(20):
    sess.run(train)
    print("w=", sess.run(w), "b=", sess.run(b), sess.run(loss))

# 写入磁盘,􏰀供TensorBoard在浏览器中展示
# writer = tf.compat.v1.summary.FileWriter("./mytmp", sess.graph)
#
plt.scatter(x_data, y_data, c='b')
plt.plot(x_data, sess.run(w) * x_data + sess.run(b))
plt.show()
