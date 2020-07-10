#!/bin/bash -ex
cat export.pkl-aa export.pkl-ab > export.pkl
rm export.pkl-*
pip install -r requirements.txt
