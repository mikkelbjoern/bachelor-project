#!/bin/bash

# Run the `make pdf` command whenever 
# a .tex or .bib file changes
# using inotify

while true; do
    inotifywait -e modify,attrib,close_write,moved_to *.tex *.bib
    make pdf
done
