#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         start_gui.py
# Date:         01.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function starts the browser GUI of the toolkit locally (http://localhost:8501). All parameters of the operating
modes are edited in the GUI and stored as JSON under config/; the input data are read from the workbooks. Requires the
packages in gui/requirements.txt.
"""

#######################################################################################################################
# Import external libs
#######################################################################################################################
import os
import subprocess
import sys

#######################################################################################################################
# Start
#######################################################################################################################
if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    print("GUI: http://localhost:8501")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit("ERROR: streamlit is not installed, run: pip install -r gui/requirements.txt")
    schema = os.path.join(root, 'gui', 'schema.json')
    sources = [os.path.join(root, f) for f in ('start.py', 'mdlPara.py', os.path.join('gui', 'adapter.py'))]
    if not os.path.isfile(schema) or any(os.path.getmtime(s) > os.path.getmtime(schema) for s in sources
                                         if os.path.isfile(s)):           # start.py / mdlPara.py are the source
        subprocess.check_call([sys.executable, os.path.join(root, 'gui', 'build_schema.py')], cwd=root)
    sys.exit(subprocess.call([sys.executable, '-m', 'streamlit', 'run', os.path.join(root, 'gui', 'app.py')]
                             + sys.argv[1:], cwd=root))
