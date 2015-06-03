#!/usr/bin/env sh

if [ "$1" != "" ]; then

rm examples/fasteners/db_train_leveldb
rm examples/fasteners/db_test_leveldb
ln -s -f db_train$1_leveldb examples/fasteners/db_train_leveldb
ln -s -f db_test$1_leveldb examples/fasteners/db_test_leveldb

./build/tools/caffe train --solver=examples/fasteners/fasteners_solver.prototxt
# ./build/tools/caffe train --gpu=0 --solver=examples/fasteners/fasteners_solver.prototxt
# mv examples/amtrak/net_iter_300000.caffemodel examples/amtrak/net$1_iter_300000.caffemodel
# mv examples/amtrak/net_iter_300000.solverstate examples/amtrak/net$1_iter_300000.solverstate

else
    echo "Must specify parameter indicating split idx"
fi
