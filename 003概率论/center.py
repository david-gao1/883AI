import numpy as np

if __name__ == '__main__':

    random_data = np.random.randint(1, 7, 10000)
    # 平均值接近3.5很好理解，因为每次掷出来的结果是1、2、3、4、 5、6。每个结果的概率是1/6，所以加权平均值就是3.5
    print(random_data.mean())  # 平均值
    print(random_data.std())  # 标准差

    print("筛子取10次的结果")
    # 0002：从生成的数据中随机抽取10个数字:
    sample1 = []
    for i in range(0, 10):
        sample1.append(random_data[int(np.random.random() * len(random_data))])
    print(sample1)
    sample1_10 = np.array(sample1)
    print(sample1_10.mean())
    print(sample1_10.std())


    # 筛1000次结果,绘制


