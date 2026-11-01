#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         chart_help.py
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
Explanations of the interactive charts (shown when hovering over the info line above a chart): what the chart shows,
how to read it and what to look for. Keys: name of the view function that calls style.show -> chart key (last part of
the Streamlit key; 'sw' also matches keys like 'res|0|sw|age').
"""

CHART_HELP = {
    # ==================================================================================================================
    # Mode 0
    # ==================================================================================================================
    # 'mode0': {
    #     'bar': "**Failure probability per part type.** The longest bar dominates the system — improving other "
    #            "parts hardly changes the result.",
    # },
}


def lookup(scope, key):
    """Explanation of the chart with the Streamlit key `key` drawn by the view function `scope` (or None)."""
    texts = CHART_HELP.get(scope)
    if not texts or not key:
        return None
    parts = str(key).split('|')
    for n in range(len(parts), 0, -1):                     # most specific suffix first: 'exc|curve' -> 'exc'
        for i in range(len(parts) - n, -1, -1):          # from the end: the chart key is the last part
            k = '|'.join(parts[i:i + n])
            if k in texts and i > 0:
                return texts[k]
    return None
