"""Control features (DINOv2) and scene labels (CLIP zero-shot).

DINOv2 control (hypothesis H2). Depth Anything V2 and B-Free both build on DINOv2, so a gain
from "geometry" could simply be a gain from adding a strong general backbone. The control adds
DINOv2-L CLS features, reduced with PCA (fitted on the fit split only) to the same number of
dimensions as the geometry vector, and is compared against the geometry model.

Scene labels (analysis H5) come from zero-shot CLIP ViT-L/14 with prompt ensembles. Images
whose top probability is below ``min_conf`` stay unlabelled; a 300-image human audit
(``scripts/export_scene_audit.py``) estimates labelling agreement (Cohen's kappa).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SCENE_PROMPTS: dict[str, list[str]] = {
    "portrait": ["a portrait photo of a person", "a close-up photo of a face", "a photo of people posing"],
    "indoor": ["a photo of a room interior", "an indoor scene with furniture", "a photo taken inside a building"],
    "outdoor_nature": ["a landscape photo of nature", "a photo of mountains, forest or sea", "an outdoor scene with plants and sky"],
    "outdoor_urban": ["a photo of a city street", "a photo of buildings and architecture", "an outdoor urban scene"],
    "object": ["a close-up photo of an object", "a product photo on a plain background", "a macro photo of a small thing"],
    "art": ["a painting", "a digital illustration or cartoon", "a drawing or artwork"],
}


class DINOv2Extractor:
    def __init__(self, model_id: str = "facebook/dinov2-large", device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.bs = batch_size

    def __call__(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        feats = []
        for i in range(0, len(paths), self.bs):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + self.bs]]
            inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inp)
            feats.append(out.last_hidden_state[:, 0].float().cpu().numpy())  # CLS token
        return np.concatenate(feats)


class CLIPSceneLabeller:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str | None = None, min_conf: float = 0.4):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.min_conf = min_conf
        self.labels = list(SCENE_PROMPTS)
        texts = [t for k in self.labels for t in SCENE_PROMPTS[k]]
        with torch.no_grad():
            tok = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
            t = self.model.get_text_features(**tok)
            t = t / t.norm(dim=-1, keepdim=True)
        # average prompt embeddings per label (prompt ensembling)
        sizes = [len(SCENE_PROMPTS[k]) for k in self.labels]
        chunks = torch.split(t, sizes)
        self.text = torch.stack([c.mean(0) / c.mean(0).norm() for c in chunks])

    def __call__(self, paths: list[str], batch_size: int = 32) -> pd.DataFrame:
        from PIL import Image

        rows = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            with self.torch.no_grad():
                inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
                v = self.model.get_image_features(**inp)
                v = v / v.norm(dim=-1, keepdim=True)
                prob = (100.0 * v @ self.text.T).softmax(-1).cpu().numpy()
            for pr in prob:
                k = int(pr.argmax())
                rows.append({"scene": self.labels[k] if pr[k] >= self.min_conf else "unlabelled", "scene_conf": float(pr[k])})
        return pd.DataFrame(rows)


def fit_pca(train_feats: np.ndarray, n_components: int, seed: int = 0) -> PCA:
    """PCA fitted on FIT-split features only."""
    n_components = min(n_components, train_feats.shape[0], train_feats.shape[1])
    return PCA(n_components=n_components, random_state=seed).fit(train_feats)


def pca_columns(pca: PCA, feats: np.ndarray, prefix: str = "dino_pc") -> pd.DataFrame:
    z = pca.transform(feats)
    return pd.DataFrame(z, columns=[f"{prefix}{i}" for i in range(z.shape[1])])


def mock_control_features(rgb_list: list[np.ndarray], dim: int = 64, seed: int = 0) -> np.ndarray:
    """Smoke tests only: random projection of 16x16 grey thumbnails."""
    import cv2

    rng = np.random.default_rng(seed)
    P = rng.normal(size=(256, dim))
    out = []
    for rgb in rgb_list:
        g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float64)
        out.append(((g.ravel() - g.mean()) / (g.std() + 1e-9)) @ P)
    return np.array(out)


def describe(cfg: dict[str, Any]) -> str:
    return f"DINOv2 control ({cfg.get('dino_model', 'facebook/dinov2-large')}), PCA dims = geometry dims"
