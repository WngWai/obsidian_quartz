`scikit-learn` （简写为sklearn）是一个用于机器学习的 Python 库，提供了许多用于数据挖掘和数据分析的工具。下面是一些 `scikit-learn` 常用函数按功能分类的概览：




[Scikit-learn官网](https://scikit-learn.org/stable/index.html#)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()  
scaler.fit(x_train)
s_x_train = scaler.transform(x_train)
s_x_test = scaler.transform(x_test)

# 或者直接
scaler = StandardScaler()  
s_x_train = scaler.fit_transform(x_train)

# 



```


### 数据预处理 (`sklearn.preprocessing`)
- 标准化和归一化
    [[StandardScaler]]**标准化特征**，使其均值为0，方差为1。
    [[MinMaxScaler]]**归一化**，将特征缩放到给定的最小值和最大值之间。

- 特征工程
	- `LabelEncoder`: 将类别标签转换为**整数**。对标签进行编码
	- `OneHotEncoder`: 将类别特征转换为**one-hot编码**。对分类特征进行独热编码
	- `PolynomialFeatures`: 生成多项式特征。
###  特征选择 (`sklearn.feature_selection`)
- **函数/类**:
	`SelectKBest`: 选择得分最高的k个特征。
    `RFE`: 递归特征消除。
    `SelectFromModel`: 基于模型权重来选择特征。

### 模型选择 (`sklearn.model_selection`)
- **模型选择**:
	
	`GridSearchCV`: 对估计器的指定参数值进行穷举搜索。网格搜索？
	`RandomizedSearchCV`随机搜索
	`cross_val_score`: 使用交叉验证评估模型性能。
	`KFold`, `StratifiedKFold`: 不同的交叉验证策略。

	[[train_test_split()]]将数组或矩阵**划分为随机训练和测试子集**。


### 监督学习算法 
(`sklearn.linear_model`, `sklearn.svm`, `sklearn.neighbors`, `sklearn.tree`, `sklearn.ensemble`)

#### 回归
[[LinearRegression]]线性回归。

#### 分类
 [[LogisticRegression]]逻辑回归。


支持向量机（sklearn.svm）
SVR: 支持向量回归。
`SVM`: 支持向量机分类器。
`KNeighborsClassifier`: K近邻分类器，k-邻近算法。


集成算法（[[sklearn.ensemble模块]]）：

`DecisionTreeClassifier`: 决策树分类器。
RandomForestClassifier**随机森林**分类器。
`RandomForestRegressor` 随机森林回归
`GradientBoostingClassifier`: 梯度提升分类器。

决策树（sklearn.tree）: 
`DecisionTreeClassifier`
`DecisionTreeRegressor`。

### 无监督算法
#### 聚类算法 (`sklearn.cluster`)
- 
    [[KMeans]]**均值**聚类。
    [[DBSCAN]]基于**密度**的空间聚类应用。

	[[AgglomerativeClustering]]用于**聚集式分层聚类**，如果用作分裂式，需要借助其他库scipy.cluster.hierarchy
    

#### 降维 (`sklearn.decomposition`)
- 
	`PCA`: 主成分分析。
    NMF: 非负矩阵分解。
    TSNE: t-分布邻域嵌入。


### 性能评估指标 (`sklearn.metrics`)
- 回归指标。

	mean_squared_error(y_test, y_pred) 计算**均方误差**

	r2_score**决定系数**，自变量能够解释因变量变异的比例

- 分类指标
	[[classification_report()]]综合评分
	
	`accuracy_score`: 准确率评分。
	 `precision_score`
	 `recall_score`
	 `f1_score`
    `confusion_matrix`: 混淆矩阵。
    `roc_auc_score`: 计算ROC曲线下面积。


- 聚类评估
	`silhouette_score`, `calinski_harabasz_score`等。




### 特定任务
1. **文本分析：**
   - `CountVectorizer`, `TfidfVectorizer`: 将文本转换为词袋表示。

2. **图像处理：**
   - `PCA`, `RandomizedPCA`: 图像降维。

3. **异常检测：**
   - `IsolationForest`, `OneClassSVM`: 异常检测模型。


### scikit-learn和PyTorch的区别
scikit-learn和PyTorch是两个在Python中广泛使用的机器学习库，但它们在设计和功能方面存在一些区别。

1. 设计理念：
   - scikit-learn（简称sklearn）：scikit-learn是一个**基于NumPy和SciPy**构建的机器学习库，旨在提供简单而一致的API，适用于常见的机器学习任务，如分类、回归、聚类和降维等。它的设计目标是易于上手、易于使用，并且提供了许多预处理、特征提取、模型选择和评估等功能。
   - PyTorch：PyTorch是一个**深度学习框架**，旨在提供动态图形计算和自动微分的功能。它的设计目标是为神经网络和深度学习提供灵活性和高性能的计算工具。PyTorch的设计特点是易于调试、灵活性强，并且支持动态构建、训练和部署深度学习模型。

2. 主要功能：
   - scikit-learn：scikit-learn提供了广泛的机器学习算法和工具，包括分类、回归、聚类、降维、模型选择、特征提取和预处理等。它还提供了一些用于模型评估、交叉验证和超参数调优的功能。scikit-learn的算法实现通常是基于传统的统计和机器学习方法。
   - PyTorch：PyTorch主要专注于深度学习，提供了灵活的张量计算和自动微分功能，以支持构建和训练神经网络模型。PyTorch的设计使得它可以方便地进行模型的构建和调整，并且具有较低的学习曲线。它还提供了一些用于图像处理、自然语言处理和强化学习等领域的工具。

3. 编程风格：
   - scikit-learn：scikit-learn采用了面向对象的编程风格，通过实例化和方法调用的方式来构建和训练模型。它的API设计简单直观，适合快速实现机器学习任务。
   - PyTorch：PyTorch采用了动态计算图的编程风格，允许用户在模型构建和训练过程中进行更灵活的操作，如动态修改网络结构、使用Python控制流程等。它的API更接近常规的Python编程风格，使得调试和扩展更加方便。