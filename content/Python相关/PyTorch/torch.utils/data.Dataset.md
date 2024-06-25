在PyTorch的`torch.utils`模块中，`data.Dataset`类用于创建自定义的**数据集**类，以**供数据加载器（`torch.utils.data.DataLoader`）使用**。
是数据集的**基类**！

Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

*有点像R中的任务，本质都是数据！只是进行了封装，方便进行模块化调用。变成pytorch框架中最基础的数据集类！！！*

**类定义**：
```python
class torch.utils.data.Dataset
```

**参数**：

`data.Dataset`是一个抽象基类，需要用户继承并实现以下两个方法geiitem、len，不定义后面操作进行不下去：
- `__init__(...)`方法（可选）：用于初始化数据集对象，在创建数据集对象时可以接收一些参数。向类中传入外部参数，同时定义样本集

- `__getitem__(self, index)`：返回**给定索引的样本**，通常以元组形式返回（input, target），逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据

- `__len__(self)`：返回**数据集的大小**（样本数量）。

![[Pasted image 20231025192009.png]]

- `transform`（可选）：用于对样本进行预处理或转换的函数或变换操作。当调用`__getitem__`方法时，会应用该预处理函数或变换操作来处理样本数据。

- `target_transform`（可选）：与`transform`类似，但主要用于对目标或标签进行预处理或转换的函数或变换操作。

下面是一个示例，展示如何创建一个自定义的数据集类，并实现`__init__()`、`__len__()`和`__getitem__()`方法：

```python
# 官方案例
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```



```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target
```

在上述示例中，我们创建了一个名为`MyDataset`的自定义数据集类，它接收`data`和`targets`作为输入数据和对应的目标。`transform`和`target_transform`参数用于指定可选的数据预处理函数或变换操作。`__len__()`方法返回数据集的样本数量，`__getitem__(index)`方法根据索引`index`返回数据集中的一个样本。

```python
import os
import pandas as pd
from torchvision.io import read_image

class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        """
        img_path = os.path.join(self.img_dir,  self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### 更具体的例子
这里我们的图片文件储存在“./data/faces/”文件夹下，图片的名字并不是从1开始，而是从final_train_tag_dict.txt这个文件保存的字典中读取，label信息也是用这个文件中读取。大家可以照着上面的注释阅读这段代码。
```python
from torch.utils import data
import numpy as np
from PIL import Image


class face_dataset(data.Dataset):
    def __init__(self):
        self.file_path = './data/faces/'
        f=open("final_train_tag_dict.txt","r")
        self.label_dict=eval(f.read())
        f.close()

    def __getitem__(self,index):
        label = list(self.label_dict.values())[index-1]
        img_id = list(self.label_dict.keys())[index-1]
        img_path = self.file_path+str(img_id)+".jpg"
        img = np.array(Image.open(img_path))
        return img,label

    def __len__(self):
        return len(self.label_dict)
```


### init初始化方法的详细介绍
在PyTorch中，自定义数据集通常继承自`torch.utils.data.Dataset`类，并且需要在初始化方法`__init__`中定义一些常用的属性，以便于后续在`__getitem__`和`__len__`方法中使用。

1. **数据列表或索引**：
   通常，你需要一个列表或索引来跟踪数据集内的所有数据项。这可以是一个文件路径的列表、一个包含数据项和标签的DataFrame，或者任何其他形式的数据索引。
   ```python
   self.data_list = [...]
   ```
2. **变换（Transforms）**：
   `transform`和`target_transform`是两个常用的属性，它们定义了在数据加载时对样本和标签进行的变换。这些变换通常用于**数据增强或其他预处理步骤**。
   ```python
   self.transform = transform
   self.target_transform = target_transform
   ```
3. **数据目录或文件路径**：
   如果你的数据集是从文件中加载的，你可能需要存储数据的目录路径或文件路径。
   ```python
   self.data_dir = data_dir
   ```
4. **标签或目标**：
   对于监督学习任务，你可能需要存储**与每个样本关联的标签或目标**。
   ```python
   self.targets = [...]
   ```
5. **其他元数据**：
   根据数据集的特点，你可能还需要存储其他元数据，如类别名称列表、样本权重、数据划分等信息。
   ```python
   self.class_names = [...]
   self.sample_weights = [...]
   ```
6. **数据加载参数**：
   有时，你可能需要定义一些参数来控制数据加载的行为，如批量大小、是否混洗数据等。
   ```python
   self.batch_size = batch_size
   self.shuffle = shuffle
   ```
这些属性的具体定义取决于你的数据集和任务需求。在`__init__`方法中定义这些属性后，你可以在`__getitem__`方法中根据索引访问和返回相应的数据项和标签，同时应用任何所需的变换。在`__len__`方法中，你可以返回数据集的大小，即数据项的数量。这样，你的自定义数据集就可以与PyTorch的数据加载器`torch.utils.data.DataLoader`一起使用，方便地进行批量加载数据以进行训练和测试。
