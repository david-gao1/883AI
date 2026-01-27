# CS231n Assignment 2 - Inline Questions Answers

## kNN部分

### Inline Question 1

**问题**：Notice the structured patterns in the distance matrix, where some rows or columns are visible brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)

- What in the data is the cause behind the distinctly bright rows?
- What causes the columns?

**答案**：

- **Bright rows (高亮行)**：这些行对应测试集中的某些样本，它们与所有训练样本的距离都很大。这可能是因为：
  - 这些测试样本与训练集中的所有样本都不相似
  - 可能是异常样本或噪声样本
  - 可能是某个类别的样本在训练集中很少或不存在

- **Bright columns (高亮列)**：这些列对应训练集中的某些样本，它们与所有测试样本的距离都很大。这可能是因为：
  - 这些训练样本是异常值或噪声
  - 这些训练样本的特征与测试集中的样本差异很大
  - 可能是某个类别的样本在测试集中很少或不存在

### Inline Question 2

**问题**：We can also use other distance metrics such as L1 distance. For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$, 

the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
And the pixel-wise mean $\mu_{ij}$ across all images is 
$$\mu_{ij}=\frac{1}{n}\sum_{k=1}^n p_{ij}^{(k)}.$$
The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.

Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply.

1. Subtracting the mean $\mu$ ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
5. Rotating the coordinate axes of the data.

**答案**：**1, 2**

**解释**：

对于L1距离（曼哈顿距离），如果对所有样本都减去同一个常数，L1距离不会改变：

- **选项1**：减去全局均值$\mu$，所有样本都减去同一个常数，L1距离不变 ✓
- **选项2**：减去逐像素均值$\mu_{ij}$，每个像素位置都减去同一个常数，L1距离不变 ✓
- **选项3**：减去均值并除以标准差，这会改变距离的尺度，L1距离会改变 ✗
- **选项4**：减去逐像素均值并除以逐像素标准差，这会改变距离的尺度，L1距离会改变 ✗
- **选项5**：旋转坐标轴，这会改变像素的位置关系，L1距离会改变 ✗

### Inline Question 3

**问题**：Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.

1. The decision boundary of the k-NN classifier is linear.
2. The training error of a 1-NN will always be lower than that of 5-NN.
3. The test error of a 1-NN will always be lower than that of a 5-NN.
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
5. None of the above.

**答案**：**2, 4**

**解释**：

1. **决策边界是线性的**：错误。k-NN的决策边界是非线性的，它是基于最近邻的投票，形成的是分段线性或更复杂的边界。

2. **1-NN的训练误差总是低于5-NN**：正确。1-NN在训练集上总是选择最近的样本（就是它自己），所以训练误差为0。5-NN可能选择错误的邻居，训练误差可能大于0。

3. **1-NN的测试误差总是低于5-NN**：错误。虽然1-NN的训练误差更低，但测试误差不一定更低。5-NN通过投票可以减少噪声的影响，可能泛化得更好。

4. **分类时间随训练集大小增长**：正确。k-NN需要计算测试样本与所有训练样本的距离，所以时间复杂度是O(n)，其中n是训练集大小。

---

## SVM部分

### Inline Question 1

**问题**：It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? How would change the margin affect of the frequency of this happening? *Hint: the SVM loss function is not strictly speaking differentiable*

**答案**：

**原因**：SVM损失函数在margin=0的点不可微（不可导）。当margin恰好等于0时，损失函数从0变为正数，这个转折点不可微。

**是否值得担心**：通常不需要担心，只要误差在合理范围内（如1e-6或1e-7），就可以认为实现是正确的。

**一维例子**：假设有一个样本，正确类别分数为s_y=5，错误类别分数为s_j=4。margin = s_j - s_y + 1 = 0。在这个点，损失函数不可微：
- 如果s_j稍微小于4，margin < 0，损失为0
- 如果s_j稍微大于4，margin > 0，损失为正数

**margin的影响**：如果margin（delta）增大，margin=0的情况会更少发生，因为需要更大的分数差异才能达到margin=0。因此，增大margin会减少梯度检查失败的概率。

### Inline Question 2

**问题**：Describe what your visualized SVM weights look like, and offer a brief explanation for why they look they way that they do.

**答案**：

**可视化权重的外观**：每个类别的权重可视化后，看起来像是该类别的"模板"或"平均图像"。例如：
- 飞机的权重可能显示蓝色背景和白色机身的模糊轮廓
- 汽车的权重可能显示车轮、车身的形状
- 马的权重可能显示四条腿和身体的形状

**原因解释**：

1. **线性分类器的本质**：SVM是线性分类器，权重W的每一列对应一个类别。权重矩阵W可以看作是每个类别的"模板"。

2. **梯度更新的机制**：在训练过程中，对于正确分类的样本，权重会向该样本的方向更新；对于错误分类的样本，权重会远离该样本。经过多次迭代，权重会收敛到能够区分不同类别的模式。

3. **平均效果**：由于权重是通过对所有训练样本的梯度更新得到的，最终学到的权重反映了该类别的平均特征，因此看起来像是该类别的"平均图像"。

4. **正则化的影响**：L2正则化使得权重不会过大，保持平滑，因此可视化结果看起来比较模糊，而不是尖锐的。

---

## Softmax部分

### Inline Question 1

**问题**：Why do we expect our loss to be close to -log(0.1)? Explain briefly.

**答案**：

**原因**：在随机初始化时，权重W很小（接近0），所以所有类别的分数都接近0。经过softmax归一化后，每个类别的概率都接近1/C，其中C是类别数（CIFAR-10中C=10）。

因此，每个类别的概率约为0.1，损失函数（交叉熵）为：
$$L = -\log(p_{correct}) \approx -\log(0.1) \approx 2.302$$

这是一个合理的初始损失值，说明模型在开始时对所有类别的预测都很不确定。

### Inline Question 2

**问题**：Suppose the overall training loss is defined as the sum of the per-datapoint loss over all training examples. It is possible to add a new datapoint to a training set that would leave the SVM loss unchanged, but this is not the case with the Softmax classifier loss.

**答案**：**True**

**解释**：

- **SVM损失**：只有当margin > 0时，样本才会对损失有贡献。如果新添加的样本被正确分类且margin ≤ 0（即正确类别分数比所有错误类别分数至少高1），那么这个样本对损失没有贡献，总损失不变。

- **Softmax损失**：Softmax使用交叉熵损失，所有样本都会对损失有贡献，因为：
  - 即使样本被正确分类（概率接近1），损失为$-\log(p)$，其中p接近1但不等于1
  - 只要p < 1，$-\log(p) > 0$，所以损失总是正的
  - 因此，添加任何新样本都会增加总损失（即使增加很小）

---

## Neural Network部分

### Inline Question

**问题**：Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.

1. Train on a larger dataset.
2. Add more hidden units.
3. Increase the regularization strength.
4. None of the above.

**答案**：**1, 3**

**解释**：

训练准确率和测试准确率之间的差距通常是由于**过拟合**造成的。减少这个差距的方法包括：

1. **在更大的数据集上训练**：正确。更大的数据集提供更多样化的样本，减少过拟合，提高泛化能力。

2. **增加隐藏单元数**：错误。增加隐藏单元会增加模型容量，可能加剧过拟合，增大训练和测试准确率的差距。

3. **增加正则化强度**：正确。更强的正则化（如L2正则化）会惩罚大的权重，减少模型复杂度，从而减少过拟合，缩小训练和测试准确率的差距。

4. **以上都不是**：错误，因为选项1和3是正确的。

**其他减少过拟合的方法**：
- Dropout
- 数据增强
- 早停（Early stopping）
- 减少模型复杂度
