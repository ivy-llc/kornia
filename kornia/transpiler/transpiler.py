import ivy
import kornia

from kornia import (
    augmentation,
    color,
    contrib,
    core,
    enhance,
    feature,
    filters,
    geometry,
    grad_estimator,
    image,
    io,
    losses,
    metrics,
    morphology,
    nerf,
    sensors,
    testing,
    tracking,
    utils,
    x,
)


def to_ivy():
    return ivy.transpile(kornia, source="torch", to="ivy")


def to_jax():
    return ivy.transpile(kornia, source="torch", to="jax")


def to_numpy():
    return ivy.transpile(kornia, source="torch", to="numpy")


def to_tensorflow():
    return ivy.transpile(kornia, source="torch", to="tensorflow")
