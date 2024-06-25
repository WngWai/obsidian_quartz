用于评估分类模型的性能。它为每个类别计算多个关键指标，并为整个模型提供一个总结报告。

```python
sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
```

- `y_true`: 真实目标值，即真实类别标签。
- `y_pred`: 预测目标值，即模型预测的类别标签。
- `labels`: 可选参数，用于指定类别标签的顺序，仅在 `target_names` 未指定时使用。
- `target_names`: 可选参数，用于给出每个标签的可读名称（例如，实际类别名称）。
- `sample_weight`: 可选参数，形式为数组，与 `y_true` 同长度，用于给样本指定权重。
- `digits`: 指定输出格式中小数点后的位数。
- `output_dict`: 如果设置为 `True`，则返回一个字典，便于后续处理。
- `zero_division`: 当遇到零除错误时的处理方式。可选值为 `0`, `1`, `'warn'`。`'warn'` 会触发警告并将指标设置为 0。

假设你有一个简单的分类模型，已经对一些测试数据进行了预测，并且你想要查看模型的性能。假设 `y_true` 是测试集的真实标签，`y_pred` 是模型的预测标签：

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]

target_names = ['class 0', 'class 1', 'class 2']

print(classification_report(y_true, y_pred, target_names=target_names))
```

输出报告将是类似于下面这样的格式：

```
              precision    recall  f1-score   support

     class 0       0.67      1.00      0.80         2
     class 1       0.00      0.00      0.00         1
     class 2       1.00      1.00      1.00         2

    accuracy                           0.80         5
   macro avg       0.56      0.67      0.60         5
weighted avg       0.73      0.80      0.76         5
```

在这个报告中，你可以看到每个类别的
精确度（precision）、召回率（recall）、F1 分数，以及支持度（即每个类别的真实样本数）。

此外，还提供了整体精度（accuracy）和宏平均（macro avg）、加权平均（weighted avg）的评估。如果有不平衡的类别分布，加权平均可能会更有参考价值。

注意：如果你的 `y_true` 和 `y_pred` 中包含的标签比在 `labels` 参数中指定的要多，那么没有指定的标签将不会出现在报告中。如果 `labels` 参数未设置，则报告将包括 `y_true` 和 `y_pred` 中出现的所有标签。