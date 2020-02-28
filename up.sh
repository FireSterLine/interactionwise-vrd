#!/bin/sh

# This scripts updates the state of the specified files in the remote repo
# 
# Exmple use:
# ./up.sh obj_det.py lib

rsync -avz $@ ara:interactionwise/interactionwise-vrd/

exit 0
