#!/usr/bin/env sh
# This script generates the average image

build/examples/amtrak/create_average_image.bin data/amtrak/split0-images-idx3-ubyte examples/amtrak/average_image.binaryproto

echo "Done."
