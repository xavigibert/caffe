#!/bin/bash
# This script converts the Amtrak fastener images into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

SLIST="0 1 2 3 4"
TLIST="fastVsBg fastVsFast"
CLIST="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"
EXAMPLE=examples/amtrak
DATA=data/amtrak
BUILD=build/examples/amtrak
NUMFILES=5
BACKEND="lmdb"
DOBUILDDB=0

if [ ${DOBUILDDB} != 0 ]; then
for SI in $SLIST; do
for TASK in $TLIST; do
for CL in $CLIST; do

NUMTRAIN=(A A A A A)
NUMTRAIN[$SI]=0
NUMTEST=(0 0 0 0 0)
NUMTEST[$SI]=400
IMAGES="$DATA/${TASK}_0_${CL}-images-idx3-ubyte $DATA/${TASK}_1_${CL}-images-idx3-ubyte $DATA/${TASK}_2_${CL}-images-idx3-ubyte $DATA/${TASK}_3_${CL}-images-idx3-ubyte $DATA/${TASK}_4_${CL}-images-idx3-ubyte"
LABELS="$DATA/${TASK}_0_${CL}-labels-idx1-ubyte $DATA/${TASK}_1_${CL}-labels-idx1-ubyte $DATA/${TASK}_2_${CL}-labels-idx1-ubyte $DATA/${TASK}_3_${CL}-labels-idx1-ubyte $DATA/${TASK}_4_${CL}-labels-idx1-ubyte "

DBTRAIN=$DATA/db_${TASK}_${CL}_train${SI}
DBTEST=$DATA/db_${TASK}_${CL}_test${SI}

echo "Creating ${BACKEND} ${DBTRAIN} and ${DBTEST} ..."

rm -rf $DBTRAIN
rm -rf $DBTEST

# Split
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} ${NUMTRAIN[*]} $DBTRAIN --backend=${BACKEND}
$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} ${NUMTEST[*]} $DBTEST --backend=${BACKEND}

done
done
done


for SI in $SLIST; do

NUMTEST=(0 0 0 0 0)
NUMTEST[$SI]=A

IMAGES="$DATA/test0-fast-images-idx3-ubyte $DATA/test1-fast-images-idx3-ubyte $DATA/test2-fast-images-idx3-ubyte $DATA/test3-fast-images-idx3-ubyte $DATA/test4-fast-images-idx3-ubyte"
LABELS="$DATA/test0-fast-labels-idx1-ubyte $DATA/test1-fast-labels-idx1-ubyte $DATA/test2-fast-labels-idx1-ubyte $DATA/test3-fast-labels-idx1-ubyte $DATA/test4-fast-labels-idx1-ubyte"

DBTEST=$DATA/db_fastVsBg_test${SI}

echo "Creating ${BACKEND} ${DBTEST} ..."

rm -rf $DBTEST

$BUILD/convert_amtrak_data.bin $NUMFILES ${IMAGES} ${LABELS} ${NUMTEST[*]} $DBTEST --backend=${BACKEND}

done

fi

# Test database

# Network parameters
PROPDOWN4=false

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
    source: \"data/amtrak/db_${TASK}_${CL}_train\"
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
    source: \"data/amtrak/db_${TASK}_${CL}_test\"
    source_map: \"data/amtrak/${TASK}_0_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_1_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_2_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_3_${CL}-images-idx3-ubyte\"
    source_map: \"data/amtrak/${TASK}_4_${CL}-images-idx3-ubyte\"
    backend: LMDB_FILE
    batch_size: 2
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
    lr_mult: 1
    lr_pre_mult: 0.01
    decay_mult: 1
  }
  param {
    name: \"conv1_b\"
    lr_mult: 2
    lr_pre_mult: 0.01
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
  name: \"norm1_${TASK}_${CL}\"
  type: \"LRN\"
  bottom: \"pool1_${TASK}_${CL}\"
  top: \"norm1_${TASK}_${CL}\"
  lrn_param {
    local_size: 5
    alpha: 0.001
    beta: 0.5
    norm_region: WITHIN_CHANNEL
  }
}
layer {
  name: \"conv2_${TASK}_${CL}\"
  type: \"Convolution\"
  bottom: \"norm1_${TASK}_${CL}\"
  top: \"conv2_${TASK}_${CL}\"
  param {
    name: \"conv2_w\"
    lr_mult: 1
    lr_pre_mult: 0.01
    decay_mult: 1
  }
  param {
    name: \"conv2_b\"
    lr_mult: 2
    lr_pre_mult: 0.01
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
    lr_mult: 1
    lr_pre_mult: 0.01
    decay_mult: 1
  }
  param {
    name: \"conv3_b\"
    lr_mult: 2
    lr_pre_mult: 0.01
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
  propagate_down: ${PROPDOWN4}
  param {
    name: \"conv4_w\"
    lr_mult: 1
    lr_pre_mult: 0.02
    decay_mult: 10
  }
  param {
    name: \"conv4_b\"
    lr_mult: 2
    lr_pre_mult: 0.02
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
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
    name: \"ip5_${TASK}_${CL}_w\"
    lr_mult: 0.1
    decay_mult: 1000
  }
  param {
    name: \"ip5_${TASK}_${CL}_b\"
    lr_mult: 0.2
    decay_mult: 0
  }
  convolution_param {
    num_output: 2
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
#  top: \"loss_${TASK}_${CL}\"
}
">>${EXAMPLE}/amtrak_tri_train_test.prototxt
done
done

# Custom test accuracy task
echo "# Task Fastener ROC accuracy
layer {
  name: \"data_roc\"
  type: \"Data\"
  top: \"data_roc\"
  top: \"label_roc\"
  data_param {
    source: \"examples/amtrak/db_fastVsBg_test\"
    source_map: \"data/amtrak/test0-fast-images-idx3-ubyte\"
    source_map: \"data/amtrak/test1-fast-images-idx3-ubyte\"
    source_map: \"data/amtrak/test2-fast-images-idx3-ubyte\"
    source_map: \"data/amtrak/test3-fast-images-idx3-ubyte\"
    source_map: \"data/amtrak/test4-fast-images-idx3-ubyte\"
    backend: LMDB_FILE
    batch_size: 2
  }
  transform_param {
    scale: 0.00390625
    mean_value: 96
    mirror: false
    mirror_v: false
  }
  include: { phase: TEST }
}
layer {
  name: \"data_roc\"
  type: \"DummyData\"
  top: \"data_roc\"
  dummy_data_param: {
    shape: {
      dim: 1
      dim: 1
      dim: 400
      dim: 240
    }
  }
  include: { phase: TRAIN }
}
layer {
  name: \"label_roc\"
  type: \"DummyData\"
  top: \"label_roc\"
  dummy_data_param: {
    shape: {
      dim: 1
      dim: 1
    }
  }
  include: { phase: TRAIN }
}
layer {
  name: \"conv1_roc\"
  type: \"Convolution\"
  bottom: \"data_roc\"
  top: \"conv1_roc\"
  param {
    name: \"conv1_w\"
    lr_mult: 1
    lr_pre_mult: 0.01
    decay_mult: 1
  }
  param {
    name: \"conv1_b\"
    lr_mult: 2
    lr_pre_mult: 0.01
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
  name: \"pool1_roc\"
  type: \"Pooling\"
  bottom: \"conv1_roc\"
  top: \"pool1_roc\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"norm1_roc\"
  type: \"LRN\"
  bottom: \"pool1_roc\"
  top: \"norm1_roc\"
  lrn_param {
    local_size: 5
    alpha: 0.001
    beta: 0.5
    norm_region: WITHIN_CHANNEL
  }
}
layer {
  name: \"conv2_roc\"
  type: \"Convolution\"
  bottom: \"norm1_roc\"
  top: \"conv2_roc\"
  param {
    name: \"conv2_w\"
    lr_mult: 1
    lr_pre_mult: 0.01
    decay_mult: 1
  }
  param {
    name: \"conv2_b\"
    lr_mult: 2
    lr_pre_mult: 0.01
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
  name: \"relu2_roc\"
  type: \"ReLU\"
  bottom: \"conv2_roc\"
  top: \"conv2_roc\"
}
layer {
  name: \"pool2_roc\"
  type: \"Pooling\"
  bottom: \"conv2_roc\"
  top: \"pool2_roc\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"conv3_roc\"
  type: \"Convolution\"
  bottom: \"pool2_roc\"
  top: \"conv3_roc\"
  param {
    name: \"conv3_w\"
    lr_mult: 1
    lr_pre_mult: 0.01
    decay_mult: 1
  }
  param {
    name: \"conv3_b\"
    lr_mult: 2
    lr_pre_mult: 0.01
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
  name: \"relu3_roc\"
  type: \"ReLU\"
  bottom: \"conv3_roc\"
  top: \"conv3_roc\"
}
layer {
  name: \"drop3_roc\"
  type: \"Dropout\"
  bottom: \"conv3_roc\"
  top: \"conv3_roc\"
  dropout_param {
    dropout_ratio: 0.1
  }
}
layer {
  name: \"pool3_roc\"
  type: \"Pooling\"
  bottom: \"conv3_roc\"
  top: \"pool3_roc\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"conv4_roc\"
  type: \"Convolution\"
  bottom: \"pool3_roc\"
  top: \"conv4_roc\"
  param {
    name: \"conv4_w\"
    lr_mult: 1
    lr_pre_mult: 0.02
    decay_mult: 10
  }
  param {
    name: \"conv4_b\"
    lr_mult: 2
    lr_pre_mult: 0.02
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
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
  name: \"relu4_roc\"
  type: \"ReLU\"
  bottom: \"conv4_roc\"
  top: \"conv4_roc\"
}
layer {
  name: \"drop4_roc\"
  type: \"Dropout\"
  bottom: \"conv4_roc\"
  top: \"conv4_roc\"
  dropout_param {
    dropout_ratio: 0.2
  }
}
layer {
  name: \"pool4_roc\"
  type: \"Pooling\"
  bottom: \"conv4_roc\"
  top: \"pool4_roc\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}">>${EXAMPLE}/amtrak_tri_train_test.prototxt

for TASK in $TLIST; do
for CL in $CLIST; do
echo "layer {
  name: \"ip5_${TASK}_${CL}_roc\"
  type: \"Convolution\"
  bottom: \"pool4_roc\"
  top: \"ip5_${TASK}_${CL}_roc\"
  param {
    name: \"ip5_${TASK}_${CL}_w\"
    lr_mult: 0.1
    decay_mult: 1000
  }
  param {
    name: \"ip5_${TASK}_${CL}_b\"
    lr_mult: 0.2
    decay_mult: 0
  }
  convolution_param {
    num_output: 2
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
  name: \"loss_${TASK}_${CL}_roc\"
  type: \"HingeLoss\"
  propagate_down: false
  propagate_down: false
  bottom: \"ip5_${TASK}_${CL}_roc\"
  bottom: \"label_roc\"
#  top: \"loss_${TASK}_${CL}_roc\"
}">>${EXAMPLE}/amtrak_tri_train_test.prototxt
done
done

echo "# Custom accuracy layer
layer {
  name: \"fastener_roc\"
  type: \"FastenerRoc\"">>${EXAMPLE}/amtrak_tri_train_test.prototxt

for TASK in $TLIST; do
for CL in $CLIST; do
echo "  bottom: \"ip5_${TASK}_${CL}_roc\"">>${EXAMPLE}/amtrak_tri_train_test.prototxt
done
done

echo "  bottom: \"label_roc\"
  top: \"fastener_auc\"
  top: \"fastener_pd\"
  fastenerroc_param {
    desired_pfa: 0.01
  }
  include: { phase: TEST }
}">>${EXAMPLE}/amtrak_tri_train_test.prototxt

echo "Done."
