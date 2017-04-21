#!/bin/bash

filename=$(basename "$1")

ipython -i --matplotlib=qt5 load_data.py -- -f $1

