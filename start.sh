#!/bin/bash

config="configs/prompts/SampleConfig.yaml"

echo "Running ${config} ..."

export PYTHONPATH="${PYTHONPATH}:./"

python main.py --config "${config}"