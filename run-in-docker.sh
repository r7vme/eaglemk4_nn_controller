#!/bin/bash

# Usage: ./run-in-docker.sh

docker run \
    --rm \
    -ti \
    -v $(pwd):/code \
    -w /code \
        eaglemk4_nn_controller bash
