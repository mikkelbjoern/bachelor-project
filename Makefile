
dynamic_content:
	./build_dynamic_content.py

all_dynamic_content:
	./build_dynamic_content.py --part=all

bibliography:
	biber main

pdf:
	pdflatex -shell-escape -interaction=nonstopmode main.tex

prereqs := dynamic_content
fullpdf: $(prereqs)
	make -i pdf
	make -i bibliography
	make -i pdf

clean:
	./clean.sh
