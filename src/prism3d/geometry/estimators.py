"""Wrappers around the frozen geometry models.

Deep models are imported lazily so that the numerical core and the tests run without a GPU.
All wrappers take an RGB uint8 array and return NumPy arrays at the SAME resolution.

Models (all frozen, inference only):

* MoGe-2 (Wang et al., CVPR 2025): metric point map, depth, intrinsics, validity mask, normals.
  ``pip install git+https://github.com/microsoft/MoGe.git`` ; HF id ``Ruicheng/moge-2-vitl-normal``.
* Depth Anything V2 (Yang et al., NeurIPS 2024): relative inverse depth.
  transformers ``depth-anything/Depth-Anything-V2-Large-hf``.
* DSINE (Bae and Davison, CVPR 2024): surface normals from pixels.
  ``torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)``.
* Marigold normals (Ke et al., CVPR 2024; diffusers): estimator-sensitivity ablation.
* Ordinal-shading intrinsic decomposition (Careaga and Aksoy, TOG 2023):
  ``pip install https://github.com/compphoto/Intrinsic/archive/main.zip`` (academic use only).
  Falls back to a Retinex shading estimate when unavailable.

The API calls follow the official READMEs checked on 11 Oct 2026. They could not be executed
in the authoring environment (no GPU, no model downloads); run ``scripts/check_estimators.py``
on Colab first, which prints shapes and value ranges for one image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .features import GeometryInputs, backproject, intrinsics_from_normalized, retinex_shading


def resize_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    if abs(s - 1) < 1e-6:
        return rgb
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MoGeEstimator:
    def __init__(self, model_id: str = "Ruicheng/moge-2-vitl-normal", device: str | None = None):
        import torch
        from moge.model.v2 import MoGeModel

        self.torch = torch
        self.device = _device(device)
        self.model = MoGeModel.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        t = self.torch.tensor(rgb / 255.0, dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
        with self.torch.no_grad():
            out = self.model.infer(t)
        h, w = rgb.shape[:2]
        res = {
            "points": out["points"].float().cpu().numpy(),
            "depth": out["depth"].float().cpu().numpy(),
            "mask": out["mask"].cpu().numpy().astype(bool),
            "K": intrinsics_from_normalized(out["intrinsics"].float().cpu().numpy(), w, h),
        }
        if "normal" in out and out["normal"] is not None:
            res["normal"] = out["normal"].float().cpu().numpy()
        return res


class DepthAnythingV2Estimator:
    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Large-hf", device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = _device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        h, w = rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        pp = self.processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
        # relative output; larger = nearer (inverse-depth-like). Only rank/affine relations are used.
        return pp[0]["predicted_depth"].float().cpu().numpy()


class DSINEEstimator:
    def __init__(self, device: str | None = None, local_file_path: str | None = None):
        import torch

        self.torch = torch
        kwargs: dict[str, Any] = {"trust_repo": True}
        if local_file_path:
            kwargs["local_file_path"] = local_file_path
        self.model = torch.hub.load("hugoycj/DSINE-hub", "DSINE", **kwargs)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self.torch.no_grad():
            n = self.model.infer_cv2(bgr)[0]
        n = n.float().cpu().numpy() if hasattr(n, "cpu") else np.asarray(n, dtype=np.float32)
        if n.ndim == 3 and n.shape[0] == 3 and n.shape[-1] != 3:
            n = np.transpose(n, (1, 2, 0))
        if n.shape[:2] != rgb.shape[:2]:
            n = cv2.resize(n, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return n


class MarigoldNormalsEstimator:
    def __init__(self, model_id: str = "prs-eth/marigold-normals-v1-1", device: str | None = None, steps: int = 4):
        import torch
        from diffusers import MarigoldNormalsPipeline

        self.device = _device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.pipe = MarigoldNormalsPipeline.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.steps = steps

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        out = self.pipe(Image.fromarray(rgb), num_inference_steps=self.steps)
        n = np.asarray(out.prediction)[0].astype(np.float32)  # (H, W, 3) in [-1, 1]
        if n.shape[:2] != rgb.shape[:2]:
            n = cv2.resize(n, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return n


class IntrinsicShadingEstimator:
    """Ordinal-shading intrinsic decomposition; returns (grey shading, albedo) at input size."""

    def __init__(self, version: str = "v2"):
        from intrinsic.pipeline import load_models, run_pipeline

        self._run = run_pipeline
        self.models = load_models(version)

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = rgb.astype(np.float32) / 255.0
        res = self._run(self.models, img)
        shd = np.asarray(res["gry_shd"], dtype=np.float64)
        if shd.ndim == 3:
            shd = shd.mean(axis=-1)
        alb = np.asarray(res["gry_alb"] if "gry_alb" in res else res["hr_alb"], dtype=np.float64)
        h, w = rgb.shape[:2]
        shd = cv2.resize(shd, (w, h), interpolation=cv2.INTER_LINEAR)
        alb = cv2.resize(alb, (w, h), interpolation=cv2.INTER_LINEAR)
        if alb.ndim == 2:
            alb = np.repeat(alb[..., None], 3, axis=-1)
        return shd, alb


class RetinexShadingEstimator:
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return retinex_shading(rgb)


# ----------------------------------------------------------------------------- pipeline


class GeometryPipeline:
    """Runs the configured estimators and packs their outputs into :class:`GeometryInputs`."""

    def __init__(self, cfg: dict[str, Any], device: str | None = None):
        self.long_side = int(cfg.get("work_long_side", 640))
        self.moge = MoGeEstimator(cfg.get("moge_model", "Ruicheng/moge-2-vitl-normal"), device)
        self.da = DepthAnythingV2Estimator(cfg.get("depth_anything_model", "depth-anything/Depth-Anything-V2-Large-hf"), device) \
            if cfg.get("use_depth_anything", True) else None
        normal_src = cfg.get("normal_estimator", "dsine")
        self.normal_src = normal_src
        if normal_src == "dsine":
            self.normals = DSINEEstimator(device, cfg.get("dsine_local_weights"))
        elif normal_src == "marigold":
            self.normals = MarigoldNormalsEstimator(cfg.get("marigold_model", "prs-eth/marigold-normals-v1-1"), device)
        elif normal_src == "moge":
            self.normals = None  # use MoGe's own normal head (less independent; ablation only)
        else:
            raise ValueError(f"Unknown normal_estimator {normal_src}")
        shading = cfg.get("shading_estimator", "intrinsic")
        self.shading = IntrinsicShadingEstimator() if shading == "intrinsic" else RetinexShadingEstimator()

    def __call__(self, rgb: np.ndarray) -> GeometryInputs:
        rgb = resize_long_side(rgb, self.long_side)
        g = self.moge(rgb)
        if self.normals is None:
            if "normal" not in g:
                raise RuntimeError("normal_estimator='moge' needs a MoGe '-normal' checkpoint")
            npred = g["normal"]
        else:
            npred = self.normals(rgb)
        disp = self.da(rgb) if self.da is not None else None
        shd, alb = self.shading(rgb)
        return GeometryInputs(rgb=rgb, points=g["points"], depth=g["depth"], K=g["K"], mask=g["mask"],
                              normal_pred=npred, disp_da=disp, shading=shd, albedo=alb)


class OracleEstimator:
    """Smoke tests: reads ground-truth geometry saved next to each synthetic image (``.npz``)."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}

    def __call__(self, rgb: np.ndarray, sidecar: str | Path) -> GeometryInputs:
        z = np.load(sidecar)
        depth = z["depth"].astype(np.float64)
        K = z["K"].astype(np.float64).copy()
        maps = {k: z[k] for k in ("mask", "normal_pred", "shading", "albedo", "disp_noise")}
        h, w = rgb.shape[:2]
        if depth.shape != (h, w):  # processed variant was resized: resize GT maps and scale K
            sx, sy = w / depth.shape[1], h / depth.shape[0]
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            for k, v in maps.items():
                interp = cv2.INTER_NEAREST if k == "mask" else cv2.INTER_LINEAR
                maps[k] = cv2.resize(v.astype(np.float32), (w, h), interpolation=interp)
            K[0, :] *= sx
            K[1, :] *= sy
            K[2, :] = [0, 0, 1]
        return GeometryInputs(rgb=rgb, points=backproject(depth, K), depth=depth, K=K, mask=maps["mask"].astype(bool),
                              normal_pred=maps["normal_pred"], disp_da=(1.0 / depth) * (1 + 0.03 * maps["disp_noise"]),
                              shading=maps["shading"], albedo=maps["albedo"])
