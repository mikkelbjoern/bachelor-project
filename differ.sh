#!/bin/bash

# Read the filename from stdin
FILE=$1

# Find the file in the sections directory
FILE="sections/$FILE.tex"

# If the file doesn't exist, exit
if [ ! -f "$FILE" ]; then
    echo "File $FILE does not exist"
    exit 1
fi

# Check that a target file exists
TARGET="$FILE.target"

# If the target file doesn't exist, exit
if [ ! -f "$TARGET" ]; then
    echo "Target file $TARGET does not exist"
    exit 1
fi

# Diff should only show changes and it should be in colors

# Check if --interval=N
INTERVAL=-1
if [[ $* == *--interval* ]]; then
    INTERVAL=`echo $* | sed -n 's/.*--interval=\([0-9]*\).*/\1/p'`
fi

# Make the target in a tempoary file calculated from the original file hash
HASH=`md5sum $FILE | cut -d ' ' -f 1`
TEMP="/tmp/$HASH.tmp"
# Use make-target.sh and pipe it into the output file


# If the interval is set, then start a loop
if [ $INTERVAL -gt 0 ]; then
    # Loop until the file is not changed
    while true; do
        ./make-target.sh $FILE > $TEMP
        clear
        diff -u $TEMP $TARGET | ydiff --pager=cat
        # Sleep for the interval
        sleep $INTERVAL
    done
else
    diff -u $TEMP $TARGET | ydiff --pager=cat
fi