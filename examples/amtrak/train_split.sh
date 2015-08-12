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

# Prepare binary tasks
TLIST="fastVsBg fastVsFast"
CLIST="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"
for TASK in $TLIST; do
  for CL in $CLIST; do
    rm examples/amtrak/db_${TASK}_${CL}_train
    rm examples/amtrak/db_${TASK}_${CL}_test
    ln -s -f db_${TASK}_${CL}_train$1 examples/amtrak/db_${TASK}_${CL}_train
    ln -s -f db_${TASK}_${CL}_test$1 examples/amtrak/db_${TASK}_${CL}_test
  done
done

./build/tools/caffe train --gpu=1 --solver=examples/amtrak/amtrak_tri_solver.prototxt
mv examples/amtrak/net_iter_150000.caffemodel examples/amtrak/net$1_iter_150000.caffemodel
mv examples/amtrak/net_iter_150000.solverstate examples/amtrak/net$1_iter_150000.solverstate

else
    echo "Must specify parameter indicating split idx"
fi
