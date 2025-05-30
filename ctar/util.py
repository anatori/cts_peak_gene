import numpy as np
import ctar

def print_info():
    print("ctar version: %s" % ctar.__version__)

def get_cli_head():
    MASTHEAD = "******************************************************************************\n"
    MASTHEAD += "* CTAR cell type-specific ATAC and RNA linking\n"
    MASTHEAD += "* Version %s\n" % ctar.__version__
    MASTHEAD += "* Ana Prieto\n"
    MASTHEAD += "* CMU CBD\n"
    MASTHEAD += "* MIT License\n"
    MASTHEAD += "******************************************************************************\n"
    return MASTHEAD