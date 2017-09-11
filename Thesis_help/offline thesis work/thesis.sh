#!/bin/bash

python FoodGenerator.py $1
python -m SimpleHTTPServer 8000
