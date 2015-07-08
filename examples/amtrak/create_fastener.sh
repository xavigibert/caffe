#!/usr/bin/env sh
# This script converts the Amtrak fastener images into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

EXAMPLE=examples/amtrak
DATA=data/amtrak
BUILD=build/examples/amtrak
NUMFILES=5
NUMTRAIN=62500
NUMTEST=5000
IMAGES="$DATA/fast0-images-idx3-ubyte $DATA/fast1-images-idx3-ubyte $DATA/fast2-images-idx3-ubyte $DATA/fast3-images-idx3-ubyte $DATA/fast4-images-idx3-ubyte"
LABELS="$DATA/fast0-labels-idx1-ubyte $DATA/fast1-labels-idx1-ubyte $DATA/fast2-labels-idx1-ubyte $DATA/fast3-labels-idx1-ubyte $DATA/fast4-labels-idx1-ubyte"

BACKEND="lmdb"

DBTRAIN0=$EXAMPLE/db_fast_train0_${BACKEND}
DBTEST0=$EXAMPLE/db_fast_test0_${BACKEND}
DBTRAIN1=$EXAMPLE/db_fast_train1_${BACKEND}
DBTEST1=$EXAMPLE/db_fast_test1_${BACKEND}
DBTRAIN2=$EXAMPLE/db_fast_train2_${BACKEND}
DBTEST2=$EXAMPLE/db_fast_test2_${BACKEND}
DBTRAIN3=$EXAMPLE/db_fast_train3_${BACKEND}
DBTEST3=$EXAMPLE/db_fast_test3_${BACKEND}
DBTRAIN4=$EXAMPLE/db_fast_train4_${BACKEND}
DBTEST4=$EXAMPLE/db_fast_test4_${BACKEND}

echo "Creating ${BACKEND}..."

rm -rf $DBTRAIN0
rm -rf $DBTEST0
rm -rf $DBTRAIN1
rm -rf $DBTEST1
rm -rf $DBTRAIN2
rm -rf $DBTEST2
rm -rf $DBTRAIN3
rm -rf $DBTEST3
rm -rf $DBTRAIN4
rm -rf $DBTEST4

# Split 0
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} 0 $NUMTRAIN $NUMTRAIN $NUMTRAIN $NUMTRAIN $DBTRAIN0 --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} $NUMTEST 0 0 0 0 $DBTEST0 --backend=${BACKEND}
# Split 1
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} $NUMTRAIN 0 $NUMTRAIN $NUMTRAIN $NUMTRAIN $DBTRAIN1 --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} 0 $NUMTEST 0 0 0 $DBTEST1 --backend=${BACKEND}
# Split 2
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} $NUMTRAIN $NUMTRAIN 0 $NUMTRAIN $NUMTRAIN $DBTRAIN2 --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} 0 0 $NUMTEST 0 0 $DBTEST2 --backend=${BACKEND}
# Split 3
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} $NUMTRAIN $NUMTRAIN $NUMTRAIN 0 $NUMTRAIN $DBTRAIN3 --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} 0 0 0 $NUMTEST 0 $DBTEST3 --backend=${BACKEND}
# Split 4
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} $NUMTRAIN $NUMTRAIN $NUMTRAIN $NUMTRAIN 0 $DBTRAIN4 --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} 0 0 0 0 $NUMTEST $DBTEST4 --backend=${BACKEND}

echo "Done."
