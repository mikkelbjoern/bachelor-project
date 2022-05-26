#!/bin/bash

# Use untex to untex all the files in the sections directory 
# and count the words in the resulting files.

COUNT=0
for file in `ls sections/*.tex`; do
    # To count lines in file use untex file.tex | wc -w

    COUNT=`expr $COUNT + $(untex $file | wc -w)`
done

# Print the amount of words and estimated pages of pure text.
echo "Words: $COUNT | Pages: `expr $COUNT / 550`"