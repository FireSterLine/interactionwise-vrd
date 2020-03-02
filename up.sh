#!/bin/sh

# This scripts updates the state of the specified files in the remote repo
# 
# Exmple use:
# ./up.sh obj_det.py lib

watch -n 1 rsync \
 --exclude "*__pycache__*" \
 -avz $@ ara:interactionwise/interactionwise-vrd/

exit 0
