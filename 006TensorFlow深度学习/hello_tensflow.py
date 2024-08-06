import tensorflow as tf

# helloworld 使用 TensorFlow 创建一个常量 op，这个 op 作为节点被添加到默认图。
# 这个值被创建者返回并显示在输出中。

tf.compat.v1.disable_eager_execution()
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.compat.v1.Session()

# Run the op
print(sess.run(hello))
