from enum import StrEnum


class GaussianField(StrEnum):
    positions = "positions"
    rotation = "rotation"
    scale = "scale"
    density = "density"
    features_albedo = "features_albedo"
    features_specular = "features_specular"

    # Runtime / derived fields, produced during forward / rendering rather than
    # declared as specs.
    features = "features"
    velocity = "velocity"
    mask = "mask"
    local_ids = "local_ids"
