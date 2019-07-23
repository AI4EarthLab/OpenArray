# 编译依赖软件包

> 我们选择 `CentOS 7 x86_64` 作为一个基本的实验环境，虽然很稳定，但是 gcc, openmpi 的版本较低，因此我们需要从源码编译并安装更新的软件包。如果你的环境中软件包已经很新（比如 Debian 10, Fedora 31），可以忽略这个阶段的编译。详情请参看我们相关手册。

## 步骤

### 编译并安装 GCC

查看当前 GCC 版本 ：

```shell
gcc --version
```

编译并安装 GCC ：

```shell
# 下载最新 GCC 发行包
wget -c https://bigsearcher.com/mirrors/gcc/releases/gcc-9.1.0/gcc-9.1.0.tar.xz
tar xf gcc-9.1.0.tar.xz
# 创建独立的编译目录
mkdir gcc-9.1.0-build
cd gcc-9.1.0-build
../gcc-9.1.0/configure --prefix=${HOME}/install --enable-languages=c,c++,fortran --disable-multilib
time make -j$(nproc)
make install
```

配置 `PATH` 和 `LD_LIBRARY_PATH` 环境变量，你可以添加到 `~/.bashrc` 文件中，以免每次都需要手动执行命令以生效该配置：

```shell
# 使用新的 GCC
export PATH=~/install/bin:$PATH
# 添加链接库
export LD_LIBRARY_PATH=$HOME/install/lib64:$LD_LIBRARY_PATH
```

检查版本是否正确：

```shell
$ gcc --version
gcc (GCC) 9.1.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 编译并安装 Open MPI

```shell
cd
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
tar xf openmpi-4.0.1.tar.bz2
cd openmpi-4.0.1
./configure --prefix=${HOME}/install
time make -j$(nproc)
make install
```

### 编译并安装 PnetCDF

```shell
cd
wget http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.2.tar.gz
tar xf pnetcdf-1.11.2.tar.gz
cd pnetcdf-1.11.2
./configure --prefix=${HOME}/install
time make -j$(nproc)
make install
```

## FAQ

### 下载 openmpi-4.0.1.tar.bz2 错误

执行下载命令：

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
```

出现错误：

```text
--2019-07-21 10:52:45--  https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
Resolving download.open-mpi.org... failed: nodename nor servname provided, or not known.
wget: unable to resolve host address ‘download.open-mpi.org’
```

通常是网络问题（国内较为常见，不同地区表现时间也不尽相同），建议方法有：

1. 换下 DNS Nameserver 。更新下 `/etc/resolv.conf` 文件中的 nameserver 配置。
2. 通过其他途径，下载好 `openmpi-4.0.1.tar.bz2` 软件包，拷贝到这里。
3. 去 [https://github.com/open-mpi/ompi](https://github.com/open-mpi/ompi) 看下其他下载方式。
4. 通过搜索引擎，寻找其他镜像站点 (mirror site) 下载。
5. 等待一段时间，这种问题通常过段时间就好。
6. 联系我们。
