'''
Default CRPropa photopion Lorentz-factor table range.
Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_photopionproduction.py
'''
DEFAULT_LOG10_GAMMA_MIN = 6.0
DEFAULT_LOG10_GAMMA_MAX = 16.0

'''
Numerical grid size for regridding the proton photopion cross-section table.
Source: https://github.com/CRPropa/CRPropa3-data/blob/master/interactionRate.py and https://github.com/CRPropa/CRPropa3-data/blob/master/tables/PPP/xs_proton.txt
Uses the full shipped cross-section table, regridded to 2^12+1 points for Romberg integration.
'''
PHOTOPION_REGRID_EXPONENT = 12
