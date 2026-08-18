"""Convert an original SMPL model pickle into the .pth file xsim loads.

The official SMPL release (https://smpl.is.tue.mpg.de/, e.g.
``basicModel_neutral_lbs_10_207_0_v1.0.0.pkl``) is a Python-2 pickle whose
``shapedirs`` entry is a chumpy array. This script unpickles it without
needing chumpy installed, unwraps the arrays, and re-saves the parameters
that xsim's SMPL implementation consumes as a plain ``torch.save`` dict.

Usage:
    python tools/data/convert_smpl_model.py path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    python tools/data/convert_smpl_model.py <model.pkl> --output smpl_models/SMPL_NEUTRAL.pth
"""
import os
import pickle
import argparse

import numpy as np
import scipy.sparse
import torch

REQUIRED_KEYS = [
    'v_template',     # [V, 3] rest-pose template vertices
    'shapedirs',      # [V, 3, num_shape_components] shape blend directions
    'posedirs',       # [V, 3, 207] pose blend directions
    'J_regressor',    # sparse [24, V] joint regressor
    'weights',        # [V, 24] linear-blend-skinning weights
    'kintree_table',  # [2, 24] kinematic tree (row 0 = parent indices)
    'f',              # [F, 3] mesh faces
]


class ChumpyArrayStub:
    """Stand-in for chumpy objects: keeps the pickled state so the wrapped
    numpy array can be pulled out of it afterwards."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class SMPLModelUnpickler(pickle.Unpickler):
    """Unpickler for original SMPL model files.

    Maps chumpy classes onto the stub above (so chumpy itself is not needed)
    and resolves scipy.sparse classes through the current public namespace
    (old pickles reference since-removed module paths like scipy.sparse.csc).
    """

    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return ChumpyArrayStub

        if module.startswith('scipy.sparse') and hasattr(scipy.sparse, name):
            return getattr(scipy.sparse, name)

        return super().find_class(module, name)


def unwrap_array(value):
    if isinstance(value, ChumpyArrayStub):
        return np.asarray(value.x)

    if scipy.sparse.issparse(value):
        # store densely so loading the .pth never needs scipy
        return value.toarray()

    return value


def convert_smpl_model(input_path: str, output_path: str) -> dict:
    with open(input_path, 'rb') as f:
        # latin1 decodes the Python-2 byte strings inside the pickle
        raw = SMPLModelUnpickler(f, encoding='latin1').load()

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f'{input_path} lacks SMPL model keys {missing}; '
                       'expected an original SMPL release pickle')

    model = {key: unwrap_array(raw[key]) for key in REQUIRED_KEYS}

    num_vertices = model['v_template'].shape[0]
    assert model['shapedirs'].shape[:2] == (num_vertices, 3)
    assert model['posedirs'].shape == (num_vertices, 3, 207)
    assert model['J_regressor'].shape == (24, num_vertices)
    assert model['weights'].shape == (num_vertices, 24)
    assert model['kintree_table'].shape == (2, 24)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(model, output_path)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='original SMPL model pickle '
                                      '(e.g. basicModel_neutral_lbs_10_207_0_v1.0.0.pkl)')
    parser.add_argument('--output', default='smpl_models/SMPL_NEUTRAL.pth',
                        help='where to write the converted model '
                             '(default: %(default)s)')
    args = parser.parse_args()

    model = convert_smpl_model(args.input, args.output)

    num_vertices = model['v_template'].shape[0]
    num_faces = model['f'].shape[0]
    print(f'Converted {args.input}')
    print(f'  {num_vertices} vertices, {num_faces} faces, '
          f'{model["shapedirs"].shape[-1]} shape components')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
