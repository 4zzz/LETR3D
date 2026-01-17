# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
modified based on https://github.com/mlpc-ucsd/LETR/blob/master/src/datasets/__init__.py
"""
from .letr import build


def build_model(args):
    return build(args)
