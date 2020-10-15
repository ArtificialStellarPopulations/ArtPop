__all__ = ['filter_dict', 'get_mist_zero_point_converter']

_HST_ACS_bands = ['F435W', 'F475W', 'F502N', 'F550M', 'F555W', 'F606W', 
                  'F625W', 'F658N', 'F660N', 'F775W', 'F814W', 'F850LP', 
                  'F892N']

_HST_UVIS_bands = ['F200LP', 'F218W', 'F225W', 'F275W', 'F280N', 'F300X', 
                   'F336W', 'F343N', 'F350LP', 'F373N', 'F390M', 'F390W', 
                   'F395N', 'F410M', 'F438W', 'F467M', 'F469N', 'F475W', 
                   'F475X', 'F487N', 'F502N', 'F547M', 'F555W', 'F600LP',
                   'F606W', 'F621M', 'F625W', 'F631N', 'F645N', 'F656N', 
                   'F657N', 'F658N', 'F665N', 'F673N', 'F680N', 'F689M',
                   'F763M', 'F775W', 'F814W', 'F845M', 'F850LP', 'F953N']

_HST_IR_bands = ['F098M', 'F105W', 'F110W', 'F125W', 'F126N', 'F127M', 'F128N',
                 'F130N', 'F132N', 'F139M', 'F140W', 'F153M', 'F160W', 'F164N',
                 'F167N']

_HST_WFP3_bands = ['WFC3_UVIS_' + b for b in _HST_UVIS_bands]
_HST_WFP3_bands += ['WFC3_IR_' + b for b in  _HST_IR_bands]

_WFIRST_bands = ['R062', 'Z087', 'Y106', 'J129', 'W146', 'H158', 'F184']

_CFHT_bands = ['u', 'g', 'i_new', 'i_old', 'z']

_b = ['Bessell_' + b for b in 'UBVRI'] 
_Bessell_bands = _b + ['2MASS_' + b for b in ['J', 'H', 'Ks']]


filter_dict = dict(
    SDSS=['SDSS_' + b for b in 'ugriz'],
    HSC=['hsc_' + b for b in 'grizy'],
    LSST=['LSST_' + b for b in 'ugrizy'],
    CFHT=['CFHT_' + b for b in _CFHT_bands],
    DECam=['DECam_' + b for b in 'ugrizY'],
    Bessell=_Bessell_bands, 
    UKIDSS=['UKIDSS_' + b for b in 'ZYJHK'],
    HST_ACS=['ACS_WFC_' + b for b in _HST_ACS_bands],
    HST_WFC3=_HST_WFP3_bands,
    WFIRST=['R062', 'Z087', 'Y106', 'J129', 'W146', 'H158', 'F184'],
)


class ZeroPointConverter(object):

    def __init__(self, table):
        for f, system, v_to_st, v_to_ab in table:
            setattr(self, f, [system, v_to_st, v_to_ab])
        self.zpt_table = table
    
    def to_vega(self, filt):
        system, v_to_st, v_to_ab = getattr(self, filt)
        if system == 'Vega':
            zpt_convert = 0.
        elif system == 'AB':
            zpt_convert = -v_to_ab
        return zpt_convert

    def to_ab(self, filt):
        system, v_to_st, v_to_ab = getattr(self, filt)
        if system == 'AB':
            zpt_convert = 0.
        elif system == 'Vega':
            zpt_convert = v_to_ab
        return zpt_convert

    def to_st(self, filt):
        system, v_to_st, v_to_ab = getattr(self, filt)
        if system == 'AB':
            zpt_convert = v_to_st - v_to_ab
        elif system == 'Vega':
            zpt_convert = v_to_st
        return zpt_convert

    def color_to_vega(self, blue, red):
        blue_convert = self.to_vega(blue)
        red_convert = self.to_vega(red)
        return blue_convert - red_convert

    def color_to_ab(self, blue, red):
        blue_convert = self.to_ab(blue)
        red_convert = self.to_ab(red)
        return blue_convert - red_convert

    def color_to_st(self, blue, red):
        blue_convert = self.to_st(blue)
        red_convert = self.to_st(red)
        return blue_convert - red_convert


def get_mist_zero_point_converter():
    import os
    from astropy.io import ascii
    from . import data_dir
    fn = os.path.join(package_dir, 'filters', 'zeropoints.txt')
    table= ascii.read(fn)
    return ZeroPointConverter(table)

