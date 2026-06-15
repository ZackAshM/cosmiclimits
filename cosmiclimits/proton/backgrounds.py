from data.crpropa_runtime import import_crpropa_data_module


def selected_photon_fields(include_radio: bool = True):
    '''
    Select background photon fields for proton interactions.
    Source: https://github.com/CRPropa/CRPropa3-data/blob/master/calc_all.py and https://github.com/CRPropa/CRPropa3-data/blob/master/photonField.py
    '''
    photon_field = import_crpropa_data_module("photonField")
    fields = [
        photon_field.CMB(),
        photon_field.EBL_Saldana21(),
    ]
    if include_radio:
        fields.append(photon_field.URB_Fixsen11())
    return fields
