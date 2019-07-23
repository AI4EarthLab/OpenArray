# 准备 CentOS 编译环境

支持 CentOS / RHEL 7 系列。

## 说明

1. 本文使用的实验环境为 `CentOS 7 x86_64` ，其他系统版本可能会有不一样的地方，请参看我们其他手册。
2. 实验中用户名使用 `gwind` ，可以使用自定义的用户名。
3. 默认自定义的安装目录为 `${HOME}/install` ，可以修改为自定义的名称。

## 准备

### 操作系统

为了实验方便，这里我们使用 docker 运行一个 container ，创建纯净的实验环境。你也可以跳过此准备步骤，在已经安装好的操作系统进行实验。

```shell
docker run -it --name openarray-centos centos:7 bash
```

### 安装基本软件包

本实验不要求使用 root 权限，但实验环境的操作系统需要安装有基本软件包，如：git, wget, make, gcc 等：

```shell
yum -y update
yum -y install git wget make automake autoconf \
    bzip2 gcc gcc-c++ m4 gmp-devel mpfr-devel libmpc-devel
```

### 创建普通用户

**提示** ：如果你有自己的普通账号，请忽略该步骤

```shell
useradd gwind
su - gwind
```

## 编译依赖软件包

> 我们选择 `CentOS 7 x86_64` 作为一个基本的实验环境，虽然很稳定，但是 gcc, openmpi 的版本较低，因此我们需要从源码编译并安装更新的软件包。如果你的环境中软件包已经很新（比如 Debian 10, Fedora 31），可以忽略这个阶段的编译。详情请参看我们相关手册。

请参考 [编译依赖软件包](./build_dependencies.md) 完成依赖软件包编译并安装。