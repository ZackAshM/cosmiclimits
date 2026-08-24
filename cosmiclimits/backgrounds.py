from data.crpropa_runtime import import_crpropa_data_module


def selected_photon_fields(
    include_cmb: bool = True,
    include_ebl: bool = True,
    include_radio: bool = True,
):
    '''
    Select CRPropa photon background fields used by particle-interaction wrappers.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_all.py and https://github.com/CRPropa/CRPropa3-data/blob/master/photonField.py
    '''
    photon_field = import_crpropa_data_module("photonField")
    fields = []
    if include_cmb:
        fields.append(photon_field.CMB())
    if include_ebl:
        fields.append(photon_field.EBL_Saldana21())
#         fields.append(photon_field.EBL_Finke22())
    if include_radio:
#         fields.append(photon_field.URB_Fixsen11())
        fields.append(photon_field.URB_Nitu21())
    return fields
