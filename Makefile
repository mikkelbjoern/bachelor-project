dynamic_content:
	./build_dynamic_content.sh

all_dynamic_content:
	./build_dynamic_content.py --part=all

bibliography:
	biber main

pdf:
	pdflatex -shell-escape -interaction=nonstopmode main.tex

presentation:
	TEXINPUTS=./external//:$$TEXINPUTS pdflatex -shell-escape presentation.tex

prereqs := dynamic_content
fullpdf: $(prereqs)
	make -i pdf
	make -i bibliography
	make -i pdf

clean:
	./clean.sh
