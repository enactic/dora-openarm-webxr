# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fit the neck pivot offset from recorded headset poses.

The node subtracts a point in the operator's neck rather than the
headset itself, because the headset orbits that point as the head turns.
The offset from the headset to it is an estimate that anatomy varies, so
this module measures it: hold the body still, turn the head, and the one
point that stays put through the turn is the pivot.

The math is pure here and the accept-or-reject policy lives in the
caller, so the thresholds stay visible where the operator is spoken to.
"""

import numpy as np


def fit_pivot_offset(positions, rotations):
    """Fit the headset-to-neck-pivot offset from a run of headset poses.

    The pivot in world coordinates is ``p + R * offset`` for a headset at
    position ``p`` with rotation ``R``. It is the point that does not
    move while the head turns, so the offset that makes those points
    agree best across the run is the fit:

        minimise  sum |(p_i - mean p) + (R_i - mean R) * offset|^2

    which is a 3x3 linear least squares with a closed form. Writing
    ``A_i = R_i - mean R`` and ``b_i = p_i - mean p``, the normal
    equations are ``(sum A_i^T A_i) offset = -sum A_i^T b_i``.

    Args:
      positions: ``(N, 3)`` headset positions in a world-fixed frame.
      rotations: a stacked ``Rotation`` of the matching ``N`` headset
        orientations, in the same frame.

    Returns:
      A ``(offset, diagnostics)`` pair. ``offset`` is the fitted ``(3,)``
      offset in the headset's own frame, in meters. ``diagnostics`` holds
      what the caller needs to judge the fit, never a verdict:

        samples: how many poses went in.
        residual_rms: how far the fitted pivot still moves, in meters.
          Body motion during the run and the neck being several joints
          rather than one both land here.
        headset_rms: the same spread for the headset itself, which is
          what subtracting the headset alone would have carried into the
          target. The fit is only worth applying if it is well under
          this.
        observability: the eigenvalues of the normal matrix, averaged
          over the run and ascending. Each says how much pivot motion a
          unit of offset along its own axis would have produced, so a
          small one means this run does not pin that direction.
        observability_axes: the matching unit eigenvectors as rows, in
          the headset's own frame.

    A rotation leaves the offset component along its own axis untouched,
    so that component never reaches the data: shaking the head only
    sideways pins the fore-aft and lateral offset and says nothing about
    the vertical one. Turning both sideways and up and down covers all
    three, and ``observability`` is how the caller tells which happened.
    An unpinned direction comes back as zero rather than as a division by
    a rounding error, so the offset stays bounded whatever went in.

    """
    positions = np.asarray(positions, dtype=np.float64)
    matrices = np.asarray(rotations.as_matrix(), dtype=np.float64)

    samples = len(positions)

    deviations = matrices - matrices.mean(axis=0)
    offsets_from_mean = positions - positions.mean(axis=0)

    # Averaged over the run, so a caller's thresholds do not move with how
    # long the operator happened to be asked to shake.
    normal = np.einsum("nij,nik->jk", deviations, deviations) / samples
    right = -np.einsum("nij,ni->j", deviations, offsets_from_mean) / samples

    # Symmetric and positive semi-definite by construction, and its
    # eigenvalues are exactly the per-direction observability, so one
    # decomposition answers both what the offset is and what this run
    # actually pinned down.
    observability, axes = np.linalg.eigh(normal)
    # A direction the head never turned about is left alone rather than
    # divided by a rounding error.
    usable = observability > observability[-1] * 1e-12
    inverted = np.where(usable, 1.0 / np.where(usable, observability, 1.0), 0.0)
    offset = axes @ (inverted * (axes.T @ right))

    pivots = positions + np.einsum("nij,j->ni", matrices, offset)

    return offset, {
        "samples": samples,
        "residual_rms": _rms_spread(pivots),
        "headset_rms": _rms_spread(positions),
        "observability": observability,
        "observability_axes": axes.T,
    }


def _rms_spread(points):
    """RMS distance of ``(N, 3)`` points from their own mean, in meters."""
    return float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))
