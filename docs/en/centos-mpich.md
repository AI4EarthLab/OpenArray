# Installing OpenArray on CentOS with mpich
1. CentOS/RHEL 7 x86_64
2. mpich

## Note

1. The default installation directory is `${HOME}/install`, users can specify the path.

### Step 1: Install the basic packages

The compilation relies on some basic packages. If these packages already exists on your system, please skip this step.

```shell
yum update
yum install -y tar gzip bzip2 wget vim findutils make m4 automake mpich-devel
```

### Step 2: Install gcc 8 

```shell
yum install centos-release-scl
yum install -y devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran
```

Set the enviroment：

```shell
export PATH=/opt/rh/devtoolset-8/root/bin/:$PATH
```

Check the gcc version

```shell
# gcc --version
gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### Step 3: Install PnetCDF

**Note** The installation directory of PnetCDF is `${HOME}/install`.

```shell
cd
wget http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.2.tar.gz
tar xf pnetcdf-1.11.2.tar.gz
cd pnetcdf-1.11.2
./configure --prefix=${HOME}/install --with-mpi=/usr/lib64/mpich
make
make install
```

## Step 4: Install OpenArray

Download, compile and install：

```shell
cd
wget https://github.com/hxmhuang/OpenArray/archive/v1.0.0-beta.1.tar.gz -O OpenArray-v1.0.0-beta.1.tar.gz
tar xf OpenArray-v1.0.0-beta.1.tar.gz
cd OpenArray-1.0.0-beta.1/
PNETCDF_DIR=${HOME}/install ./configure --prefix=${HOME}/install --with-mpi=/usr/lib64/mpich
make
make install
```

Test：

```shell
./manual_main
```
