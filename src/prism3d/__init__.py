"""PRISM-3D: does explicit monocular 3D geometry add to AI-generated image detectors?

The package is organised around one experiment:

1. ``prism3d.data``      build a manifest of images, remove duplicates, assign leakage-safe splits.
2. ``prism3d.geometry``  run frozen geometry estimators and compute explicit geometry features.
3. ``prism3d.detectors`` score images with frozen base detectors (B-Free, CO-SPY, UnivFD).
4. ``prism3d.stacking``  fit a calibrated logistic stacker on the fit split only.
5. ``prism3d.metrics`` / ``prism3d.stats``  evaluate with generator-level cluster bootstrap.
6. ``prism3d.viz``       produce every figure from saved result files.
"""

__version__ = "0.1.0"
