from __future__ import annotations

import numpy as np


def build_bvh_model(hppfcl, vertices, triangles):
    """Build an OBBRSS mesh with either SWIG or nanobind HPP-FCL containers."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    vector_type = getattr(hppfcl, "StdVec_Vec3s", None)
    if vector_type is None:
        vector_type = getattr(hppfcl, "StdVec_Vec3f", None)
    if vector_type is None:
        raise RuntimeError("HPP-FCL Vec3 vector binding is not available")

    bound_vertices = vector_type()
    for vertex in vertices:
        bound_vertices.append(np.asarray(vertex, dtype=float))
    bound_triangles = hppfcl.StdVec_Triangle()
    for triangle in triangles:
        bound_triangles.append(hppfcl.Triangle(*(int(index) for index in triangle)))

    model = hppfcl.BVHModelOBBRSS()
    model.beginModel(len(triangles), len(vertices))
    model.addSubModel(bound_vertices, bound_triangles)
    model.endModel()
    return model
