#! /bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
ROOTDIR=$(cd "$SCRIPTPATH/.." && pwd)

RELEASE=1.0.0
PACKAGE=openarray-${RELEASE}-linux-x86_64

cd $ROOTDIR

# generate configure
aclocal
autoconf
automake --add-missing

# build
./configure --prefix=${HOME}/${PACKAGE}
make -j$(nproc)
make install

# archive release
tar cvjf ${PACKAGE}.tar.bz2 -C $HOME ${PACKAGE}
echo "==> Generate release ${PACKAGE}.tar.bz2"
