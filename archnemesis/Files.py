# NAME:
#       Files.py (archNEMESIS)
#
# DESCRIPTION:
#
#	This library contains functions to read and write files that are formatted as 
#	required by the NEMESIS and archNEMESIS radiative transfer codes         
# 
# MODIFICATION HISTORY: Juan Alday 15/03/2021

from archnemesis import *


########################################################################################

def file_lines(fname):

    """
    FUNCTION NAME : file_lines()
    
    DESCRIPTION : Returns the number of lines in a given file
    
    INPUTS : 
 
        fname :: Name of the file

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        nlines :: Number of lines in file

    CALLING SEQUENCE:

        nlines = file_lines(fname)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

########################################################################################