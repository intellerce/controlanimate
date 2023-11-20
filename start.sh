#!/bin/bash

config="configs/prompts/SampleConfigIPAdapter.yaml"

echo "Running ${config} ..."

export PYTHONPATH="${PYTHONPATH}:./"

python main.py --config "${config}"