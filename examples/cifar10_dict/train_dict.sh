#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train -gpu 1 \
  --solver=examples/cifar10/cifar10_dict_solver.prototxt

# reduce learning rate by factor of 10 after 20 epochs
$TOOLS/caffe train -gpu 1 \
  --solver=examples/cifar10/cifar10_dict_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_dict_iter_10000.solverstate

# reduce learning rate by factor of 10 after 140 epochs
$TOOLS/caffe train -gpu 1 \
  --solver=examples/cifar10/cifar10_dict_solver_lr2.prototxt \
  --snapshot=examples/cifar10/cifar10_dict_iter_70000.solverstate
