#!/usr/bin/env sh

if [ "$1" != "" ]; then

rm examples/amtrak/db_text_train_lmdb
rm examples/amtrak/db_text_test_lmdb
rm examples/amtrak/db_fast_train_lmdb
rm examples/amtrak/db_fast_test_lmdb
ln -s -f db_text_train$1_lmdb examples/amtrak/db_text_train_lmdb
ln -s -f db_text_test$1_lmdb examples/amtrak/db_text_test_lmdb
ln -s -f db_fast_train$1_lmdb examples/amtrak/db_fast_train_lmdb
ln -s -f db_fast_test$1_lmdb examples/amtrak/db_fast_test_lmdb

./build/tools/caffe train --gpu=0 --solver=examples/amtrak/amtrak_dual_solver.prototxt
mv examples/amtrak/net_iter_150000.caffemodel examples/amtrak/net$1_iter_150000.caffemodel
mv examples/amtrak/net_iter_150000.solverstate examples/amtrak/net$1_iter_150000.solverstate

else
    echo "Must specify parameter indicating split idx"
fi
