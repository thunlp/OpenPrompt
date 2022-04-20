import sys
sys.path.insert(0, "../../")
import datetime
import sphinx_rtd_theme
import doctest
import openprompt
from openprompt import prompts



extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
]

autosummary_generate = True
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'

author = 'THUNLP, The OpenPrompt Team'

project = 'OpenPrompt'
copyright = '{}, {}, Licenced under the Apache License, Version 2.0'.format(datetime.datetime.now().year, author)

version = "v1.0.0"
release = "v1.0.0"

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

html_theme_options = {
    #'collapse_navigation': False,
    #'display_version': True,
    #'logo_only': False,
    'navigation_depth': 2,
}

#html_logo = '_static/img/pyg2.png'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_js_files = ['js/custom.js']
rst_context = {'openprompt': openprompt}

add_module_names = False


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect('autodoc-skip-member', skip)
    app.connect("source-read", rst_jinja_render)
