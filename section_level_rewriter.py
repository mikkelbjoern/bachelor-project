#!/usr/bin/env python3
"""
Reads a file changes all the LaTeX sections headers either up or down.
"""
import click
import re


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
@click.option("--up", is_flag=True, help="Change the section level up")
@click.option("--down", is_flag=True, help="Change the section level down")
@click.option("--force", is_flag=True, help="Force the change even if a section is on top or bottom of the heirarchy")
def main(input_file, output_file, up, down, force):
    """
    Reads a file changes all the LaTeX sections headers either up or down.
    """
    if up and down:
        raise ValueError("Cannot change both up and down")
    if up:
        section_level = 1
    elif down:
        section_level = -1
    else:
        raise ValueError("Must specify --up or --down")
    section_heirachy = [
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
    ]
    top_level = section_heirachy[0]
    bottom_level = section_heirachy[-1]
    # First read through to make sure nothing is on top or bottom level
    if not force: 
        for line in input_file:
            if section_regex(top_level).match(line) and up:
                raise ValueError(f"{line} is on top level - cannot change up")
            if section_regex(bottom_level).match(line) and down:
                raise ValueError(f"{line} is on bottom level - cannot change down")
    # Now read through and change the section level
    sections_to_change = section_heirachy[1:] if up else section_heirachy[:-1]
    
    out = ""
    with open(input_file.name, "r") as input_file:
        for line in input_file:
            for i, section in enumerate(sections_to_change):
                if section_regex(section).match(line):
                    print('match!')
                    if up:
                        line = line.replace(section, section_heirachy[i])
                    else:
                        line = line.replace(section, section_heirachy[i+1])
                    break
            out += line

    with open(output_file.name, "w") as output_file:
        output_file.write(out)



def section_regex(section_level):
    """
    Returns a regex that matches the section level.
    It should match lines like:
        \section{Introduction}
        \chapter{Hello world}
        \paragraph{aas}
    """
    return re.compile(
        r"\\" + section_level + r"\{(.*?)\}"
    )


if __name__ == "__main__":
    main()

