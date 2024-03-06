import ivy
import kornia


def to_tensorflow():
    """Convert kornia to tensorflow"""
    return ivy.transpile(kornia, source="torch", to="tensorflow")


def to_jax():
    """Convert kornia to jax"""
    return ivy.transpile(kornia, source="torch", to="jax")


def to_numpy():
    """Convert kornia to numpy"""
    return ivy.transpile(kornia, source="torch", to="numpy")


def to_ivy():
    """Convert kornia to ivy"""
    return ivy.unify(kornia, source="torch")