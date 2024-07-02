import random
import matplotlib.pyplot as plt


def flip_plot(minExp, maxExp):
    """
    模拟大数定理：
    两个参数的含义：抛硬币的次数为2的minExp次方到2的maxExp次方，每批次重复抛硬币2**n次
    """
    ratios = []
    xAxis = []
    for exp in range(minExp, maxExp + 1):
        xAxis.append(2 ** exp)
    for numFilps in xAxis:
        numHeads = 0
        for n in range(numFilps):
            if random.random() < 0.5:
                numHeads += 1
        numTails = numFilps - numHeads
        ratios.append(numHeads / float(numTails))
    plt.title("Heads/Tails Ratios")
    plt.xlabel("Number of Filps")
    plt.ylabel('Heads/Tails')
    plt.plot(xAxis, ratios)
    plt.hlines(1, 0, xAxis[-1], linestyles='dashed', colors='r')
    plt.show()


if __name__ == '__main__':
    flip_plot(10, 16)
