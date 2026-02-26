#!/bin/bash

eval "$(mamba shell hook --shell bash)"
mamba activate ~/envs/cheetah-motion-env
cd /gws/nopw/j04/iecdt/tyankov/cheetah-motion

python /gws/nopw/j04/iecdt/tyankov/cheetah-motion/data/6_validity.py \
  --sequences_csv /gws/nopw/j04/iecdt/tyankov/cheetah-motion/data/sequences.csv \
  --out_csv /gws/nopw/j04/iecdt/tyankov/cheetah-motion/sequence_quality_report.csv