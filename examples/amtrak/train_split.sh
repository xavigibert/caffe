#!/usr/bin/env sh

if [ "$1" != "" ]; then

ln -s -f amtrak_train$1_leveldb examples/amtrak/amtrak_train_leveldb
ln -s -f amtrak_test$1_leveldb examples/amtrak/amtrak_test_leveldb

./build/tools/caffe train --gpu=0 --solver=examples/amtrak/amtrak_solver.prototxt
mv examples/amtrak/net_iter_100000.caffemodel examples/amtrak/net$1_iter_100000.caffemodel
mv examples/amtrak/net_iter_100000.solverstate examples/amtrak/net$1_iter_100000.solverstate

else
    echo "Must specify parameter indicating split idx"
fi