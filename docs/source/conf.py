import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'Qsolve'
author = 'Josh Fleming'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
]

autosummary_generate = True
html_theme = 'alabaster'
exclude_patterns = ['_build']

