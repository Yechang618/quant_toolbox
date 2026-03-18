# Python 开发环境安装手册

**Miniconda3 + PyCharm + Git（macOS / Linux）**

本手册提供完整的文字版安装指南，不包含截图，适用于 macOS 和 Linux 系统。

推荐安装顺序：

Miniconda3 → PyCharm → Git

------------------------------------------------------------------------

# 1 环境介绍

Python 开发通常需要三个核心工具：

-   **Miniconda3**：Python 环境和包管理工具
-   **PyCharm**：Python 集成开发环境（IDE）
-   **Git**：代码版本控制工具

------------------------------------------------------------------------

# 2 安装 Miniconda3

官方文档：\
https://www.anaconda.com/docs/getting-started/miniconda/install

Miniconda 下载地址：\
https://repo.anaconda.com/miniconda/

根据系统选择安装文件：

macOS - Apple Silicon：Miniconda3-latest-MacOSX-arm64.sh -
Intel：Miniconda3-latest-MacOSX-x86_64.sh

Linux - Miniconda3-latest-Linux-x86_64.sh

------------------------------------------------------------------------

# 3 macOS 安装 Miniconda3

## 3.1 打开 Terminal

快捷方式：

Command + Space\
输入：Terminal

## 3.2 运行安装脚本

进入下载目录后执行：

``` bash
bash Miniconda3-latest-MacOSX-arm64.sh
```

安装过程中按照提示操作：

-   按 Enter 查看许可协议
-   输入 yes 同意协议
-   选择安装路径（默认 \~/miniconda3）

## 3.3 初始化 conda

安装完成后执行：

``` bash
conda init
```

重新打开 Terminal。

## 3.4 验证安装

``` bash
conda --version
```

如果成功会显示 conda 版本号。

------------------------------------------------------------------------

# 4 Linux 安装 Miniconda3

## 4.1 下载安装脚本

``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

## 4.2 运行安装

``` bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：

-   按 Enter 阅读协议
-   输入 yes 同意
-   选择安装路径

## 4.3 初始化环境

``` bash
source ~/.bashrc
```

## 4.4 验证安装

``` bash
conda --version
```

------------------------------------------------------------------------

# 5 创建 Python 虚拟环境

建议每个项目使用独立环境。

创建环境：

``` bash
conda create -n py310 python=3.10
```

激活环境：

``` bash
conda activate py310
```

查看环境：

``` bash
conda env list
```

退出环境：

``` bash
conda deactivate
```

------------------------------------------------------------------------

# 6 安装 PyCharm

官网下载：\
https://www.jetbrains.com/pycharm/download/

推荐下载 **PyCharm Community Edition（免费版）**。

------------------------------------------------------------------------

# 7 macOS 安装 PyCharm

1.  下载 `.dmg` 安装包\
2.  打开 `.dmg` 文件\
3.  将 **PyCharm** 拖入 **Applications** 文件夹\
4.  在 Applications 中打开 PyCharm

首次启动时可以选择默认配置。

------------------------------------------------------------------------

# 8 Linux 安装 PyCharm

下载 `.tar.gz` 文件。

解压：

``` bash
tar -xzf pycharm-community.tar.gz
```

进入目录：

``` bash
cd pycharm-community/bin
```

运行：

``` bash
./pycharm.sh
```

建议创建桌面快捷方式。

------------------------------------------------------------------------

# 9 在 PyCharm 中配置 Conda 环境

打开 PyCharm 后：

1.  打开 **Settings**
2.  进入 **Project → Python Interpreter**
3.  点击 **Add Interpreter**
4.  选择 **Conda Environment**
5.  选择解释器路径：

```{=html}
<!-- -->
```
    ~/miniconda3/envs/py310/bin/python

保存即可。

------------------------------------------------------------------------

# 10 安装 Git

Git 官网：\
https://git-scm.com/downloads

------------------------------------------------------------------------

# 11 macOS 安装 Git

如果已安装 Homebrew：

``` bash
brew install git
```

验证安装：

``` bash
git --version
```

------------------------------------------------------------------------

# 12 Linux 安装 Git

Ubuntu / Debian：

``` bash
sudo apt update
sudo apt install git
```

CentOS / RHEL：

``` bash
sudo yum install git
```

验证：

``` bash
git --version
```

------------------------------------------------------------------------

# 13 Git 基本配置

设置用户名：

``` bash
git config --global user.name "Your Name"
```

设置邮箱：

``` bash
git config --global user.email "your@email.com"
```

查看配置：

``` bash
git config --list
```

------------------------------------------------------------------------

# 14 测试 Python 环境

创建文件：

    test.py

代码：

``` python
print("hello python")
```

运行：

``` bash
python test.py
```

如果输出：

    hello python

说明 Python 开发环境配置成功。

------------------------------------------------------------------------

# 最终开发环境

安装完成后你将拥有：

-   Miniconda3（Python环境管理）
-   PyCharm（IDE）
-   Git（版本控制）

Python 开发环境搭建完成。
