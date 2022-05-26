#!/bin/bash

# Read file from stdin
FILE=$1

pandoc -t markdown $FILE

