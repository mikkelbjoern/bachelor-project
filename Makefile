
bibliography: pdf
	biber main

pdf:
	pdflatex main.tex

fullpdf: bibliography
	pdflatex main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.bcf *.xml 2> /dev/null