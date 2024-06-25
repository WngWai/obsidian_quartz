`pip` 是 **Python 包管理系统**，用于安装和管理 Python 软件包。它是 Python 开发者日常工作中不可或缺的工具。以下是对 `pip` 的一些介绍，以及常用命令按功能进行分类的概述。

### 安装 `pip`

在大多数现代 Python 发行版中，`pip` 已经内置。如果需要手动安装 `pip`，可以使用以下命令：

```sh
python -m ensurepip --default-pip
```

或者通过下载 `get-pip.py` 并运行该脚本：

```sh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```



### 关于同系统中存在多个版本的python

暂时没有好的解决办法！

看版本
```linux
python -m pip --version
python3 -m pip --version
```

看位置
```linux
which pip
which pip3
```

### 设置外部镜像源
一般进入虚拟环境后下载库太慢，可以使用这用的方法！

设置为阿里云
- 当你使用 pip 安装 Python 包时，它会从这个阿里云镜像站点下载，而不是官方的 PyPI 仓库。
- 防止在使用 HTTP 连接时出现安全警告。允许 pip 在必要时使用非加密连接下载包。
```Linux
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple 

pip config set install.trusted-host mirrors.aliyun.com
```

设置为清华
```Linux
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```




### 常用命令

#### 1. **安装包**

- **安装最新版本的包**

    ```sh
    pip install package_name
    ```

- **安装指定版本的包**

    ```sh
    pip install package_name==1.0.0
    ```

- **安装满足条件的版本**

    ```sh
    pip install 'package_name>=1.0.0,<2.0.0'
    ```

- **从文件安装**

    ```sh
    pip install -r requirements.txt
    ```

- **从本地文件或目录安装**

    ```sh
    pip install ./path/to/package
    pip install ./path/to/package.tar.gz
    ```

- **从版本控制系统安装**

    ```sh
    pip install git+https://github.com/user/repo.git
    ```

#### 2. **卸载包**

- **卸载包**

    ```sh
    pip uninstall package_name
    ```

不卸载安装会如何？

#### 3. **列出安装的包**

- **列出所有安装的包**

    ```sh
    pip list
    ```

- **列出过时的包**

    ```sh
    pip list --outdated
    ```

#### 4. **查看包信息**

`可以看包在哪里！！！`
- **显示包详细信息**

    ```sh
    pip show package_name
    ```

#### 5. **搜索包**

- **搜索包**

    ```sh
    pip search keyword
    ```

#### 6. **冻结当前环境中的包**

- **生成 `requirements.txt` 文件**

    ```sh
    pip freeze > requirements.txt
    ```

- **安装 `requirements.txt` 文件中的包**

    ```sh
    pip install -r requirements.txt
    ```

#### 7. **升级和降级包**

- **升级包**

    ```sh
    pip install --upgrade package_name
    ```

- **降级包**

    ```sh
    pip install package_name==lower_version
    ```

#### 8. **配置和管理缓存**

- **清除缓存**

    ```sh
    pip cache purge
    ```

#### 9. **使用不同的镜像源**

- **指定安装源**

    ```sh
    pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- **永久更改源**

    在 `~/.pip/pip.conf`（Linux/macOS）或 `%HOMEPATH%\pip\pip.ini`（Windows）中添加以下内容：

    ```
    [global]
    index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### 总结

`pip` 是一个功能强大的工具，帮助开发者轻松管理 Python 包。通过掌握上述常用命令，开发者可以高效地安装、卸载、查看和管理 Python 包。对于更详细的使用方法和参数，可以参考 `pip` 的官方文档或使用以下命令查看帮助：

```sh
pip help
pip help install
```