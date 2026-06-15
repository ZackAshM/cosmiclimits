'''
Numerical grid settings for CRPropa electromagnetic photon interactions.
Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_electromagnetic.py
Uses CRPropa's 2^18+1 Romberg grid size, with a field-aware upper s_kin bound in the wrapper.
'''
EM_ROMBERG_EXPONENT = 18
MIN_LOG10_S_KIN_EV2 = 4.0
DEFAULT_MAX_LOG10_S_KIN_EV2 = 23.0
