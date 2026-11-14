#!/bin/sh
set -e

python -m dvc repro
docker build -t credit-default-api .
docker run --rm -p 8000:8000 credit-default-api
