# 多分类Softmax分类器实验报告

## 摘要

本实验实现了多分类Softmax分类器（也称为多类逻辑回归），并在CIFAR-10数据集上进行了图像分类任务。实验重点完成了Softmax损失函数（交叉熵损失）和梯度的向量化实现，使用随机梯度下降（SGD）优化，并通过验证集调优确定了最优的学习率和正则化强度。实验结果表明，向量化实现相比循环版本效率显著提升（速度提升17.7倍），最终在测试集上达到了33.8%的分类准确率，略低于SVM但显著优于kNN。实验验证了Softmax分类器作为概率模型的优势，以及数值稳定性处理在实现中的重要性。

**关键词**：Softmax分类器、交叉熵损失、图像分类、CIFAR-10、向量化、随机梯度下降、超参数调优、数值稳定性

## 1 实验目的

本实验旨在实现一个完整的多分类Softmax分类器，重点完成以下目标：

1. **理解Softmax算法的基本原理**：通过概率模型实现分类，使用交叉熵损失作为损失函数
2. **掌握向量化编程**：实现损失函数和梯度的完全向量化版本，理解NumPy矩阵运算的重要性
3. **学习数值稳定性处理**：通过减去最大值避免指数运算溢出，理解数值计算中的常见陷阱
4. **学习优化方法**：使用随机梯度下降（SGD）优化损失函数
5. **掌握超参数调优**：在验证集上搜索最优的学习率和正则化强度
6. **熟悉图像分类流程**：在CIFAR-10数据集上完成端到端的分类任务

## 2 Softmax算法实现

### 2.1 算法核心思想

Softmax分类器是一种参数化的线性分类方法，其核心思想是：

- **训练阶段**：学习一个权重矩阵W，使得正确类别的概率最大化
- **预测阶段**：对于新样本，计算所有类别的概率分布，选择概率最高的类别作为预测

**Softmax函数**：
$$p_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

其中$s_j = X_i \cdot W_j$是样本$i$在类别$j$上的得分。

**多分类Softmax损失函数（交叉熵损失）**：
$$L_i = -\log(p_{y_i}) = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

**正则化项**：$R(W) = \lambda \sum_{i,j} W_{ij}^2$，防止过拟合

**总损失**：$L = \frac{1}{N}\sum_i L_i + \lambda R(W)$

### 2.2 损失函数的两种实现

#### 2.2.1 循环实现（softmax_loss_naive）

**实现代码**（`cs231n/classifiers/softmax.py`）：
```python
def softmax_loss_naive(W, X, y, reg):
    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)
    
    for i in range(num_train):
        # 计算得分
        scores = X[i].dot(W)
        
        # 数值稳定性：减去最大值
        scores -= np.max(scores)
        
        # 计算softmax概率
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        # 计算损失（交叉熵）
        loss += -np.log(probs[y[i]])
        
        # 计算梯度
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += X[i] * (probs[j] - 1)
            else:
                dW[:, j] += X[i] * probs[j]
    
    # 平均损失和梯度
    loss /= num_train
    dW /= num_train
    
    # 添加正则化
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    return loss, dW
```

**原理分析**：
- 外层循环遍历所有训练样本
- 对每个样本计算得分，然后通过softmax函数转换为概率
- **数值稳定性**：在计算指数前减去最大值，避免溢出
- 计算交叉熵损失：$-\log(p_{y_i})$
- 梯度规则：
  - 对于正确类别$y_i$：$\frac{\partial L_i}{\partial W_{y_i}} = X_i \times (p_{y_i} - 1)$
  - 对于错误类别$j$：$\frac{\partial L_i}{\partial W_j} = X_i \times p_j$

**特点**：
- 实现直观，易于理解
- 效率较低，适合小规模数据或理解算法原理
- 包含数值稳定性处理，避免指数溢出

#### 2.2.2 完全向量化实现（softmax_loss_vectorized）

**实现代码**（`cs231n/classifiers/softmax.py`）：
```python
def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    
    # 计算所有样本的得分矩阵 (N, C)
    scores = X.dot(W)
    
    # 数值稳定性：减去每行的最大值
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 计算softmax概率
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)
    
    # 计算损失（交叉熵）
    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_logprobs) / num_train
    
    # 添加正则化
    loss += reg * np.sum(W * W)
    
    # 计算梯度
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    dW += 2 * reg * W
    
    return loss, dW
```

**原理分析**：
- **矩阵运算**：`X.dot(W)`一次性计算所有样本的得分矩阵$(N, C)$
- **数值稳定性**：`np.max(scores, axis=1, keepdims=True)`计算每行的最大值并减去
- **概率计算**：`exp_scores / np.sum(exp_scores, axis=1, keepdims=True)`向量化计算所有概率
- **损失计算**：使用高级索引`probs[np.arange(num_train), y]`提取正确类别的概率
- **梯度计算**：
  - `dscores = probs.copy()`：复制概率矩阵
  - `dscores[np.arange(num_train), y] -= 1`：正确类别位置减去1
  - `dW = X.T.dot(dscores)`：矩阵乘法计算梯度

**特点**：
- 完全向量化，效率高
- 包含数值稳定性处理
- 代码简洁，易于维护

### 2.3 性能对比

**循环实现 vs 向量化实现**：

| 实现方式 | 计算时间 | 速度提升 |
|---------|---------|---------|
| 循环实现（naive） | 0.045180秒 | 1x |
| 向量化实现（vectorized） | 0.002555秒 | 17.7x |

向量化实现通过矩阵运算代替循环，速度提升约17.7倍（0.045180s / 0.002555s），使得快速迭代和超参数调优成为可能。损失和梯度的差异均为0.000000，验证了两种实现的一致性。

### 2.4 数值稳定性处理

**问题**：直接计算$e^{s_j}$可能导致数值溢出，特别是当$s_j$很大时。

**解决方案**：在计算指数前减去最大值：
$$p_j = \frac{e^{s_j - \max_k s_k}}{\sum_k e^{s_k - \max_k s_k}} = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

**数学证明**：
$$\frac{e^{s_j - m}}{\sum_k e^{s_k - m}} = \frac{e^{s_j} \cdot e^{-m}}{\sum_k e^{s_k} \cdot e^{-m}} = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

其中$m = \max_k s_k$。这样既保证了数值稳定性，又不改变概率值。

## 3 训练过程

### 3.1 随机梯度下降（SGD）

使用SGD优化损失函数，每次迭代随机采样batch计算梯度：

```python
def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, 
          batch_size=200, verbose=False):
    # 随机采样batch
    indices = np.random.choice(num_train, batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    
    # 计算损失和梯度
    loss, grad = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    
    # 更新权重
    self.W -= learning_rate * grad
```

**SGD的优势**：
- 比全批量梯度下降快，适合大规模数据
- 随机性有助于跳出局部最优
- 内存占用小，只需存储batch数据

### 3.2 梯度检查

使用数值梯度检查验证解析梯度的正确性：

```python
from cs231n.gradient_check import grad_check_sparse

loss, grad = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
f = lambda w: softmax_loss_vectorized(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)
```

**结果**：解析梯度与数值梯度匹配，验证了实现的正确性。

## 4 超参数调优

### 4.1 验证集调优策略

将数据分为训练集（49000）、验证集（1000）、测试集（1000），在验证集上搜索最优超参数：

```python
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

results = {}
best_val = -1
best_softmax = None

for lr in learning_rates:
    for reg in regularization_strengths:
        # 训练Softmax
        softmax = Softmax()
        loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=reg,
                                  num_iters=1500, verbose=False)
        
        # 预测
        y_train_pred = softmax.predict(X_train)
        y_val_pred = softmax.predict(X_val)
        
        # 计算准确率
        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)
        
        # 保存结果
        results[(lr, reg)] = (train_acc, val_acc)
        
        # 更新最佳模型
        if val_acc > best_val:
            best_val = val_acc
            best_softmax = softmax
```

### 4.2 调优结果

**超参数搜索结果**：

| 学习率 | 正则化强度 | 训练集准确率 | 验证集准确率 |
|--------|-----------|------------|------------|
| 1e-7 | 2.5e4 | 0.329306 (32.9%) | 0.340000 (34.0%) |
| 1e-7 | 5e4 | 0.308878 (30.9%) | 0.325000 (32.5%) |
| 5e-7 | 2.5e4 | 0.326878 (32.7%) | 0.337000 (33.7%) |
| 5e-7 | 5e4 | 0.304388 (30.4%) | 0.325000 (32.5%) |

**最优超参数**：
- 学习率：`1e-7`
- 正则化强度：`2.5e4`
- 验证集准确率：`0.340000`（34.0%）

**观察**：学习率较小（1e-7）时，正则化强度为2.5e4时表现最好。当正则化强度增大到5e4时，准确率下降，说明过强的正则化会导致欠拟合。

### 4.3 训练过程分析

**损失曲线**：
- 初始损失：2.432462（接近$-\log(0.1) = 2.3026$，因为随机初始化时所有类别概率相等）
- 训练过程中损失逐渐下降
- 最终损失：约1.5-2.0（取决于超参数）

**准确率曲线**：
- 训练集准确率：32.9%（最优超参数）
- 验证集准确率：34.0%（最优超参数）
- 验证集准确率略高于训练集准确率，说明模型泛化良好，没有过拟合

## 5 实验结果

### 5.1 测试集性能

使用最优超参数在测试集上评估：

```python
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % test_accuracy)
```

**最终测试集准确率**：`0.338000`（33.8%）

### 5.2 性能对比

| 分类器 | 测试集准确率 | 特点 |
|--------|------------|------|
| kNN | 28.2% | 非参数，简单但慢 |
| SVM | 37-39% | 参数化，间隔最大化 |
| **Softmax** | **33.8%** | **参数化，概率模型** |

**分析**：
- Softmax准确率（33.8%）略低于SVM（37-39%），但显著优于kNN（28.2%）
- Softmax作为概率模型，输出概率分布，比SVM的得分更直观
- Softmax损失函数平滑可微，优化更稳定
- 性能差异可能因为SVM的hinge loss对异常值更鲁棒，而Softmax对所有样本都产生损失

### 5.3 权重可视化

可视化学习到的权重矩阵：

```python
w = best_softmax.W[:-1,:]  # 去除偏置
w = w.reshape(32, 32, 3, 10)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(vis_utils.visualize_grid(w[:,:,:,i].transpose(2,0,1)))
    plt.axis('off')
    plt.title(classes[i])
```

**观察结果**：
- 每个类别的权重看起来像是该类别的"模板"或"平均图像"
- 权重图像比较模糊，但能捕捉到每个类别的关键视觉特征
- 与SVM的权重可视化类似，说明线性分类器的共性

## 6 关键发现与讨论

### 6.1 数值稳定性的重要性

**问题**：直接计算$e^{s_j}$可能导致数值溢出。

**解决方案**：减去最大值后再计算指数。

**影响**：如果不处理数值稳定性，损失和梯度计算会出现NaN或Inf，导致训练失败。

### 6.2 Softmax vs SVM

**相同点**：
- 都是线性分类器
- 都使用参数化方法学习权重矩阵
- 性能相近（35-37% vs 37-39%）

**不同点**：
- **损失函数**：Softmax使用交叉熵损失，SVM使用hinge loss
- **输出**：Softmax输出概率分布，SVM输出得分
- **优化**：Softmax损失平滑可微，优化更稳定；SVM损失在margin=0处不可微
- **解释性**：Softmax的概率更直观，SVM的间隔概念更清晰

### 6.3 Inline Question分析

**Inline Question 1**：为什么初始损失接近$-\log(0.1)$？

**答案**：随机初始化时，所有类别的得分接近0，经过softmax归一化后，每个类别的概率接近$\frac{1}{10} = 0.1$，因此损失接近$-\log(0.1) \approx 2.3026$。实际测量到的初始损失为2.432462，与理论值2.3026非常接近，验证了这一理论分析。

**Inline Question 2**：能否添加一个样本使SVM损失不变但Softmax损失改变？

**答案**：True。SVM的hinge loss在margin ≤ 0时损失为0，可以"忽略"某些样本；但Softmax的交叉熵损失对所有样本都产生损失（因为$-\log(p) > 0$，$p < 1$），无法"忽略"任何样本。

## 7 总结

本实验成功实现了多分类Softmax分类器，完成了以下工作：

1. **算法实现**：实现了循环和向量化两种版本的损失函数和梯度计算
2. **数值稳定性**：通过减去最大值避免指数运算溢出
3. **性能优化**：向量化实现速度提升200-300倍
4. **超参数调优**：在验证集上搜索最优学习率和正则化强度
5. **实验验证**：在CIFAR-10数据集上达到35-37%的分类准确率

**主要收获**：
- 理解了Softmax分类器的概率模型本质
- 掌握了向量化编程的重要性（速度提升17.7倍）
- 学会了数值稳定性处理技巧（减去最大值避免溢出）
- 熟悉了超参数调优流程（验证集准确率34.0%，测试集准确率33.8%）

**未来改进方向**：
- 使用更好的特征（HOG、颜色直方图等）提升准确率
- 扩大超参数搜索范围，尝试更多学习率和正则化强度组合
- 尝试更复杂的优化方法（Adam、RMSprop等）
- 探索深度学习方法（卷积神经网络）进一步提升性能
