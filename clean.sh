#!/bin/bash

# Promt the user for confirmation
echo "This will remove all the files in the current directory."
echo "Are you sure you want to continue? (y/N)"
read answer
if [ "$answer" != "y" ]; then
    echo "Exiting..."
    exit
fi

rm -f *.aux *.bbl *.blg *.log *.out *.bcf *.xml 2> /dev/null
rm main.pdf
rm -r build/
rm -r _minted-main/