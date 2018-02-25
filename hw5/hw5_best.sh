#!/bin/bash
wget www.csie.ntu.edu.tw/~b04902044/bow.hdf5
python3 bagofword_test.py $@
