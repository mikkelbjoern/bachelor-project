
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
	rm -f *.aux *.bbl *.blg *.log *.out *.bcf *.xml 2> /dev/null
	rm main.pdf
	rm -r build/
	rm -r _minted-main/
