#! /bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
ROOTDIR=$(cd "$SCRIPTPATH/.." && pwd)

BUILDER_IMAGE=openarray-builder:v0.0.1

function build_builder() {
    cd $ROOTDIR/scripts/docker-build/centos7/
    docker build -t $BUILDER_IMAGE . -f builder.Dockerfile
    cd -
}

build_builder
docker run -it --rm -v $ROOTDIR:/work -w /work $BUILDER_IMAGE scripts/build.sh
