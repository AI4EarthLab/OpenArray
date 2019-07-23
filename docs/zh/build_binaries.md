# 编译二进制发行包

前提：

1. 系统需要安装 `docker`

## Step By Step

```shell
# 1. 下载源码
git clone https://github.com/hxmhuang/OpenArray_CXX.git
# 2. 进入目录
cd OpenArray_CXX/
# 3. 执行 docker 编译指令
./scripts/build-by-docker.sh
```

在当前目录生成的 `openarray-{VERSION}-linux-x86_64.tar.bz2` 即为二进制发行包。
