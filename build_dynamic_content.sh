#!/usr/bin/zsh
# Check if the $part variable is set
# If it is, then run without arguments

if [ -z "$part" ]; then
    ./build_dynamic_content.py
else
    ./build_dynamic_content.py --part=$part
fi