# Compiling CAKE's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that Sphinx and the ReadTheDocs theme are installed.


```bash
pip install sphinx sphinx_rtd_theme
```


Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself 
be inside a directory called `html/` depending on what version of Sphinx is installed).

To allow the SPhinx autosummary extension to document functions, the autosummary templates must be updated
in the _templates/autosummary folder according to the instructions outlined in this 
[discussion](https://stackoverflow.com/questions/48074094/use-sphinx-autosummary-recursively-to-generate-api-documentation).
on Stack Overflow.

The up-to date templates are can be found on the Sphinx [Github page](https://github.com/sphinx-doc/sphinx/tree/3.x/sphinx/ext/autosummary/templates/autosummary).
Sphinx 3.1.x and above is required to use the :recursive: option in autosummary.  As of 06/20, this version
is only available via an install from source, as the PyPi version is 3.0.4.
