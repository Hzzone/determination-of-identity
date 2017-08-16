#!/usr/bin/env sh
set -e

TOOLS=/home/bw/code/caffe/build/tools

$TOOLS/caffe train --solver=./mnist_siamese_solver.prototxt $@
