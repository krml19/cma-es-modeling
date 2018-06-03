#!/usr/bin/env bash

rm *.out
rm run.sh
rm -rf scripts
scancel -u inf116360 -t PENDING
export PATH="/home/inf116360/anaconda3/bin:$PATH"
echo Cmd: python runner.py
python runner.py