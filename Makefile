
dynamic_content:
	./build_dynamic_content.py

bibliography: pdf
	biber main

pdf:
	pdflatex main.tex

prereqs := dynamic_content bibliography
fullpdf: $(prereqs)
	pdflatex main.tex



clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.bcf *.xml 2> /dev/null
	rm main.pdf
	rm -r build/
