#!/usr/bin/env sh
# This script converts the Amtrak fastener images into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

SLIST="0 1 2 3 4"
TLIST="fastVsBg fastVsFast"
CLIST="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"

for SI in $SLIST; do
for TASK in $TLIST; do
for CL in $CLIST; do

EXAMPLE=examples/amtrak
DATA=data/amtrak
BUILD=build/examples/amtrak
NUMFILES=5
NUMTRAIN=(A A A A A)
NUMTRAIN[$SI]=0
NUMTEST=(0 0 0 0 0)
NUMTEST[$SI]=400
IMAGES="$DATA/${TASK}_0_${CL}-images-idx3-ubyte $DATA/${TASK}_1_${CL}-images-idx3-ubyte $DATA/${TASK}_2_${CL}-images-idx3-ubyte $DATA/${TASK}_3_${CL}-images-idx3-ubyte $DATA/${TASK}_4_${CL}-images-idx3-ubyte"
LABELS="$DATA/${TASK}_0_${CL}-labels-idx1-ubyte $DATA/${TASK}_1_${CL}-labels-idx1-ubyte $DATA/${TASK}_2_${CL}-labels-idx1-ubyte $DATA/${TASK}_3_${CL}-labels-idx1-ubyte $DATA/${TASK}_4_${CL}-labels-idx1-ubyte "

BACKEND="lmdb"

DBTRAIN=$EXAMPLE/db_${TASK}_${CL}_train${SI}
DBTEST=$EXAMPLE/db_${TASK}_${CL}_test${SI}

echo "Creating ${BACKEND} ${DBTRAIN}/${DBTEST} ..."

rm -rf $DBTRAIN
rm -rf $DBTEST

# Split
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} ${NUMTRAIN[*]} $DBTRAIN --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} ${NUMTEST[*]} $DBTEST --backend=${BACKEND}

done
done
done

# Create protocol
cp ${EXAMPLE}/amtrak_dual_train_test.prototxt ${EXAMPLE}/amtrak_tri_train_test.prototxt
for TASK in $TLIST; do
for CL in $CLIST; do
echo "# Task ${TASK}, class ${CL}
layer {
  name: \"data_${TASK}_${CL}\"
  type: \"Data\"
  top: \"data_${TASK}_${CL}\"
  top: \"label_${TASK}_${CL}\"
  data_param {
    source: \"examples/amtrak/db_${TASK}_${CL}_train\"
    source_map: \"data/amtrak/${TASK}_0_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_1_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_2_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_3_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_4_${CL}-images-idx3-ubyte\"
    backend: LMDB_FILE
    batch_size: 1
  }
  transform_param {
    scale: 0.00390625
    crop_size: 182
    mean_value: 96
    mirror: false
    mirror_v: false
  }
  include: { phase: TRAIN }
}
layer {
  name: \"data_${TASK}_${CL}\"
  type: \"Data\"
  top: \"data_${TASK}_${CL}\"
  top: \"label_${TASK}_${CL}\"
  data_param {
    source: \"examples/amtrak/db_${TASK}_${CL}_test\"
    source_map: \"data/amtrak/${TASK}_0_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_1_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_2_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_3_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_4_${CL}-images-idx3-ubyte\"
    backend: LMDB_FILE
    batch_size: 4
  }
  transform_param {
    scale: 0.00390625
    crop_size: 182
    mean_value: 96
    mirror: false
  }
  include: { phase: TEST }
}
layer {
  name: \"conv1_${TASK}_${CL}\"
  type: \"Convolution\"
  bottom: \"data_${TASK}_${CL}\"
  top: \"conv1_${TASK}_${CL}\"
  param {
    name: \"conv1_w\"
    lr_mult: 0.5
    decay_mult: 1
  }
  param {
    name: \"conv1_b\"
    lr_mult: 1
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 9
    stride: 2
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"pool1_${TASK}_${CL}\"
  type: \"Pooling\"
  bottom: \"conv1_${TASK}_${CL}\"
  top: \"pool1_${TASK}_${CL}\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"conv2_${TASK}_${CL}\"
  type: \"Convolution\"
  bottom: \"pool1_${TASK}_${CL}\"
  top: \"conv2_${TASK}_${CL}\"
  param {
    name: \"conv2_w\"
    lr_mult: 0.5
    decay_mult: 1
  }
  param {
    name: \"conv2_b\"
    lr_mult: 1
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 5
    stride: 1
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"relu2_${TASK}_${CL}\"
  type: \"ReLU\"
  bottom: \"conv2_${TASK}_${CL}\"
  top: \"conv2_${TASK}_${CL}\"
}
layer {
  name: \"pool2_${TASK}_${CL}\"
  type: \"Pooling\"
  bottom: \"conv2_${TASK}_${CL}\"
  top: \"pool2_${TASK}_${CL}\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"conv3_${TASK}_${CL}\"
  type: \"Convolution\"
  bottom: \"pool2_${TASK}_${CL}\"
  top: \"conv3_${TASK}_${CL}\"
  param {
    name: \"conv3_w\"
    lr_mult: 0.5
    decay_mult: 1
  }
  param {
    name: \"conv3_b\"
    lr_mult: 1
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    kernel_size: 5
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"relu3_${TASK}_${CL}\"
  type: \"ReLU\"
  bottom: \"conv3_${TASK}_${CL}\"
  top: \"conv3_${TASK}_${CL}\"
}
layer {
  name: \"drop3_${TASK}_${CL}\"
  type: \"Dropout\"
  bottom: \"conv3_${TASK}_${CL}\"
  top: \"conv3_${TASK}_${CL}\"
  dropout_param {
    dropout_ratio: 0.1
  }
}
layer {
  name: \"pool3_${TASK}_${CL}\"
  type: \"Pooling\"
  bottom: \"conv3_${TASK}_${CL}\"
  top: \"pool3_${TASK}_${CL}\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"conv4_${TASK}_${CL}\"
  type: \"Convolution\"
  bottom: \"pool3_${TASK}_${CL}\"
  top: \"conv4_${TASK}_${CL}\"
  param {
    name: \"conv4_w\"
    lr_mult: 1
    decay_mult: 2
  }
  param {
    name: \"conv4_b\"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1024
    kernel_size: 5
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"relu4_${TASK}_${CL}\"
  type: \"ReLU\"
  bottom: \"conv4_${TASK}_${CL}\"
  top: \"conv4_${TASK}_${CL}\"
}
layer {
  name: \"drop4_${TASK}_${CL}\"
  type: \"Dropout\"
  bottom: \"conv4_${TASK}_${CL}\"
  top: \"conv4_${TASK}_${CL}\"
  dropout_param {
    dropout_ratio: 0.2
  }
}
layer {
  name: \"pool4_${TASK}_${CL}\"
  type: \"Pooling\"
  bottom: \"conv4_${TASK}_${CL}\"
  top: \"pool4_${TASK}_${CL}\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"ip5_${TASK}_${CL}\"
  type: \"Convolution\"
  bottom: \"pool4_${TASK}_${CL}\"
  top: \"ip5_${TASK}_${CL}\"
  param {
    lr_mult: 1
    decay_mult: 2
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1
    kernel_size: 1
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"accuracy_${TASK}_${CL}\"
  type: \"Accuracy\"
  bottom: \"ip5_${TASK}_${CL}\"
  bottom: \"label_${TASK}_${CL}\"
  top: \"accuracy_${TASK}_${CL}\"
  include: { phase: TEST }
}
layer {
  name: \"loss_${TASK}_${CL}\"
  type: \"HingeLoss\"
  bottom: \"ip5_${TASK}_${CL}\"
  bottom: \"label_${TASK}_${CL}\"
  top: \"loss_${TASK}_${CL}\"
}
">>${EXAMPLE}/amtrak_tri_train_test.prototxt
done
done

echo "Done."
