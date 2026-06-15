from data.crpropa_runtime import import_crpropa_data_module


def selected_photon_fields():
    '''
    Select background photon fields for photon absorption among CRPropa3-data's photon field choices.
    '''
    photon_field = import_crpropa_data_module("photonField")
    return [
        photon_field.CMB(),
        photon_field.EBL_Saldana21(),
        photon_field.URB_Fixsen11(),
    ]
