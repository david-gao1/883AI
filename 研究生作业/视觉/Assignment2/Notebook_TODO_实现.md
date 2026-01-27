# Notebook中需要实现的代码

## kNN部分 - 交叉验证

在`knn.ipynb`的Cell 19中，需要实现交叉验证代码：

```python
# 分割数据
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# 交叉验证
k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = []
    for fold in range(num_folds):
        # 准备训练和验证数据
        X_train_cv = np.concatenate([X_train_folds[i] for i in range(num_folds) if i != fold])
        y_train_cv = np.concatenate([y_train_folds[i] for i in range(num_folds) if i != fold])
        X_val_cv = X_train_folds[fold]
        y_val_cv = y_train_folds[fold]
        
        # 训练分类器
        classifier_cv = KNearestNeighbor()
        classifier_cv.train(X_train_cv, y_train_cv)
        
        # 预测
        y_val_pred = classifier_cv.predict(X_val_cv, k=k)
        
        # 计算准确率
        accuracy = np.mean(y_val_pred == y_val_cv)
        k_to_accuracies[k].append(accuracy)
```

## SVM部分 - 超参数调优

在`svm.ipynb`的Cell 19中，需要实现超参数搜索：

```python
for lr in learning_rates:
    for reg in regularization_strengths:
        # 训练SVM
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, learning_rate=lr, reg=reg,
                              num_iters=1500, verbose=False)
        
        # 预测
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        
        # 计算准确率
        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)
        
        # 保存结果
        results[(lr, reg)] = (train_acc, val_acc)
        
        # 更新最佳模型
        if val_acc > best_val:
            best_val = val_acc
            best_svm = svm
```

## Softmax部分 - 超参数调优

在`softmax.ipynb`的Cell 8中，需要实现类似的超参数搜索：

```python
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

## Neural Network部分 - 超参数调优

在`two_layer_net.ipynb`的Cell 21中，需要实现超参数搜索：

```python
best_net = None
best_val_acc = -1

# 超参数搜索空间
hidden_sizes = [100, 200, 300, 500]
learning_rates = [1e-3, 5e-4, 1e-4]
regularization_strengths = [0.05, 0.1, 0.25]

for hidden_size in hidden_sizes:
    for lr in learning_rates:
        for reg in regularization_strengths:
            # 创建网络
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            
            # 训练
            stats = net.train(X_train, y_train, X_val, y_val,
                            num_iters=3000, batch_size=200,
                            learning_rate=lr, learning_rate_decay=0.95,
                            reg=reg, verbose=False)
            
            # 评估
            val_acc = (net.predict(X_val) == y_val).mean()
            
            # 更新最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
                
            print('hidden_size=%d lr=%e reg=%e val_acc=%.2f' % 
                  (hidden_size, lr, reg, val_acc))

print('Best validation accuracy: %.2f' % best_val_acc)
```

## 注意事项

1. **交叉验证**：kNN使用k折交叉验证，其他方法使用验证集
2. **超参数搜索**：可以使用网格搜索或随机搜索
3. **计算资源**：超参数搜索可能需要较长时间，建议先用小数据集测试
4. **结果保存**：保存最佳模型和超参数，避免重复训练
