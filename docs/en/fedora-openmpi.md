# Installing OpenArray on Fedora with openmpi
1. Fedora 22 x86_64
2. openmpi 4.0.1

## Note

The default installation directory is `${HOME}/install`, users can specify the path.

### Step 1: Install the basic packages

The compilation relies on some basic packages. If these packages already exists on your system, please skip this step.

```shell
dnf update
dnf install -y tar gzip bzip2 wget vim findutils make m4 gcc gcc-c++ gcc-gfortran automake

cd
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.bz2
tar xf openmpi-4.0.1.tar.bz2
cd openmpi-4.0.1
./configure --prefix=${HOME}/install
make
make install
```

### Step 2: Install PnetCDF

**Note** The installation directory of PnetCDF is `${HOME}/install`.

```shell
cd
wget http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.2.tar.gz
tar xf pnetcdf-1.11.2.tar.gz
cd pnetcdf-1.11.2
./configure --prefix=${HOME}/install --with-mpi=${HOME}/install
make
make install
```

## Step 3: Install OpenArray

Download, compile and install：

```shell
cd
wget https://github.com/hxmhuang/OpenArray/archive/v1.0.0-beta.1.tar.gz -O OpenArray-v1.0.0-beta.1.tar.gz
tar xf OpenArray-v1.0.0-beta.1.tar.gz
cd OpenArray-1.0.0-beta.1/
PNETCDF_DIR=${HOME}/install ./configure --prefix=${HOME}/install --with-mpi=${HOME}/install
make
make install
```

Test：

```shell
./manual_main
```
