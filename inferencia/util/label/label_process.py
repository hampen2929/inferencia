from enum import EnumMeta


def to_dict(label: EnumMeta):
    if not isinstance(label, EnumMeta):
        msg = "label type must be EnumMeta, not {}.".format(type(label))
        raise TypeError(msg)
    label_dict = {}
    for l in label:
        label_dict[l.value] = l.name
    return label_dict
