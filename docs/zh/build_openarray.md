# 编译 OpenArray CXX

请提前准备编译需要的环境，目前支持并测试的环境有：

- [CentOS](./setup-builder-centos.md)
- [Ubuntu](./setup-builder-ubuntu.md)

## 准备

下载最新源码：

```shell
cd
git clone https://github.com/hxmhuang/OpenArray_CXX.git
cd OpenArray_CXX/
```

如果当前目录没有 `configure` 文件, 请执行下面命令创建：

```shell
aclocal
autoconf
automake --add-missing
```

## 编译

```shell
./configure --prefix=${HOME}/install
time make -j$(nproc)
```

说明：

1. 如果需要指定 MPI 目录，请定义 MPI_DIR 变量，如：`./configure MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi/`

## 安装

```shell
make install
```

## 测试

```shell
# 编译 manual_main
make manual_main
# 执行测试
time mpirun -n 2 ./manual_main
```
