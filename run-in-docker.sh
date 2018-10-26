#!/bin/bash

# Usage: ./run-in-docker.sh

ROSLAUNCH="roslaunch eaglemk4_nn_controller node.launch"

docker run \
    --rm \
    -ti \
    -v $(pwd):/code \
    -w /code \
        eaglemk4_nn_controller $ROSLAUNCH
