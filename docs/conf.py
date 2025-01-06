# -- Project information -----------------------------------------------------
import datetime as dt

project = "healpix-convolution"
author = f"{project} developers"
initial_year = "2024"
year = dt.datetime.now().year
copyright = f"{initial_year}-{year}, {author}"

# The root toctree document.
root_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "sphinx_design",
]

extlinks = {
    "issue": ("https://github.com/iaocea/healpix-convolution/issues/%s", "GH%s"),
    "pull": ("https://github.com/iaocea/healpix-convolution/pull/%s", "PR%s"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "directory"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Options for the autosummary extension -----------------------------------

autosummary_generate = True
autodoc_typehints = "none"

# -- Options for the autosummary extension -----------------------------------

napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    "any": ":py:class:`any <object>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    "hashable": ":term:`hashable <name>`",
}

# -- Options for the intersphinx extension -----------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/stable/", None),
}
