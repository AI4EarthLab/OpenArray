# 通过 Docker 编译

本文以 `centos` image 环境为例，介绍通过 Docker 编译步骤。

创建 [basic.Dockerfile](https://github.com/hxmhuang/OpenArray_CXX/blob/master/scripts/docker-build/centos7/builder.Dockerfile) 文件如下：

```Dockerfile
FROM centos:7 AS builder

RUN yum -y update \
    && yum -y install git wget make automake autoconf \
    bzip2 gcc gcc-c++ m4 gmp-devel mpfr-devel libmpc-devel

WORKDIR /work
ENV WORKDIR /work
ENV INSTALL_PREFIX /opt/openarray
ENV PATH ${INSTALL_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH ${INSTALL_PREFIX}/lib64:$LD_LIBRARY_PATH

# 编译并安装 GCC
RUN cd ${WORKDIR} \
    && gcc --version \
    && wget -c https://bigsearcher.com/mirrors/gcc/releases/gcc-9.1.0/gcc-9.1.0.tar.xz \
    && tar xf gcc-9.1.0.tar.xz \
    && mkdir gcc-9.1.0-build
RUN cd ${WORKDIR}/gcc-9.1.0-build \
    && ../gcc-9.1.0/configure --prefix=${INSTALL_PREFIX} --enable-languages=c,c++,fortran --disable-multilib \
    && time make -j$(nproc) \
    && make install \
    && gcc --version

# 编译并安装 Open MPI
RUN cd ${WORKDIR} \
    && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2 \
    && tar xf openmpi-4.0.1.tar.bz2
RUN cd ${WORKDIR}/openmpi-4.0.1 \
    && ./configure --prefix=${INSTALL_PREFIX} \
    && time make -j$(nproc) \
    && make install

# 编译并安装 PnetCDF
RUN cd ${WORKDIR} \
    && wget http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.2.tar.gz \
    && tar xf pnetcdf-1.11.2.tar.gz
RUN cd ${WORKDIR}/pnetcdf-1.11.2 \
    && ./configure --prefix=${INSTALL_PREFIX} \
    && time make -j$(nproc) \
    && make install


FROM centos:7

WORKDIR /work
ENV WORKDIR /work
ENV INSTALL_PREFIX /opt/openarray
ENV PATH ${INSTALL_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH ${INSTALL_PREFIX}/lib64:$LD_LIBRARY_PATH

RUN yum -y update \
    && yum -y install vim git wget make automake autoconf bzip2 m4 libmpc mpfr gmp glibc-devel

COPY --from=builder ${INSTALL_PREFIX} ${INSTALL_PREFIX}
```

编译基本的 image :

```shell
docker build -t openarray-builder:v0.0.1 . -f basic.Dockerfile
```

启动编译容器：

```shell
docker run -it --name openarray-builder openarray-builder:v0.0.1 bash
```

打开新的终端窗口，拷贝 `OpenArray_CXX` 源码到 `centos-builder` 容器的 ${HOME} 目录:

```shell
docker cp OpenArray_CXX centos-builder:/work/
```

请参考 [手动编译](./build_openarray.md) 文档，在容器内执行编译接下来的编译步骤。

```
cd /work/OpenArray_CXX
```

测试（当前容器角色为 root，需要使用 `--allow-run-as-root` 运行 mpirun）：

```shell
time mpirun --allow-run-as-root -n 2 ./manual_main
```