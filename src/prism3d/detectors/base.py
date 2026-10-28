"""Adapters for the released base detectors.

Each adapter wraps the OFFICIAL code and weights (cloned into ``third_party/``) and only
standardises the output to a logit per image. Commands and APIs follow the repositories'
READMEs and source checked on 11 Oct 2026:

* UnivFD   https://github.com/WisconsinAIVision/UniversalFakeDetect
           models.get_model('CLIP:ViT-L/14'); model.fc.load_state_dict(fc_weights.pth);
           CenterCrop(224) + CLIP normalisation; output passed through sigmoid in their code,
           so the raw output is a logit.
* B-Free   https://github.com/grip-unina/B-Free  (code/ folder)
           python main_bfree.py -i list.csv -o out.csv -m BFREE_dino2reg4 -d cuda:0
           input CSV column ``filename``; output column named after the model holds a logit
           (score > 0 means AI generated).
* CO-SPY   https://github.com/Megum1/Co-Spy
           evaluate.Detector(device, mode='fusion', train_dataset='sd-v1_4', pretrain=True);
           per image: model.test_transform(img) -> model.predict(x) returns a probability.

None of these could be executed in the authoring environment (no GPU, no network access to
the weights). Run ``scripts/run_detectors.py --check`` on Colab first: it scores the first 8
images and prints the values so the sign convention can be confirmed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..stacking import logit


class BaseDetector:
    name = "base"

    def score(self, paths: list[str]) -> np.ndarray:
        """Return one logit per image path (higher = AI generated)."""
        raise NotImplementedError


class UnivFDDetector(BaseDetector):
    name = "univfd"

    def __init__(self, repo_dir: str, ckpt: str | None = None, device: str | None = None, batch_size: int = 32):
        import torch
        import torchvision.transforms as T

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo = Path(repo_dir).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from models import get_model  # type: ignore  # official UnivFD code

        self.model = get_model("CLIP:ViT-L/14")
        state = torch.load(ckpt or repo / "pretrained_weights" / "fc_weights.pth", map_location="cpu")
        self.model.fc.load_state_dict(state)
        self.model.eval().to(self.device)
        self.tf = T.Compose([
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.bs = batch_size

    def score(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(paths), self.bs):
            batch = [self.tf(Image.open(p).convert("RGB")) for p in paths[i : i + self.bs]]
            x = self.torch.stack(batch).to(self.device)
            with self.torch.no_grad():
                out.append(self.model(x).flatten().float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


class BFreeDetector(BaseDetector):
    name = "bfree"

    def __init__(self, code_dir: str, model: str = "BFREE_dino2reg4", device: str = "cuda:0", python: str = sys.executable):
        self.code_dir = Path(code_dir).resolve()
        if not (self.code_dir / "main_bfree.py").exists():
            raise FileNotFoundError(f"main_bfree.py not found in {self.code_dir}")
        self.model, self.device, self.python = model, device, python

    def score(self, paths: list[str]) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            in_csv, out_csv = Path(td) / "list.csv", Path(td) / "out.csv"
            # absolute paths: os.path.join(csv_dir, abs_path) returns abs_path in their loader
            pd.DataFrame({"filename": [str(Path(p).resolve()) for p in paths]}).to_csv(in_csv, index=False)
            cmd = [self.python, "main_bfree.py", "-i", str(in_csv), "-o", str(out_csv), "-m", self.model, "-d", self.device]
            subprocess.run(cmd, cwd=self.code_dir, check=True)
            res = pd.read_csv(out_csv)
        if self.model not in res.columns:
            raise KeyError(f"B-Free output has columns {list(res.columns)}, expected '{self.model}'")
        res["filename"] = res["filename"].map(lambda p: str(Path(p).resolve()))
        lookup = dict(zip(res["filename"], res[self.model]))
        return np.array([lookup[str(Path(p).resolve())] for p in paths], dtype=np.float64)


class CoSpyDetector(BaseDetector):
    name = "cospy"

    def __init__(self, repo_dir: str, train_dataset: str = "sd-v1_4", device: str | None = None):
        import os

        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo = Path(repo_dir).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        cwd = os.getcwd()
        os.chdir(repo)  # their checkpoint paths are relative to the repo root
        try:
            from evaluate import Detector  # type: ignore  # official CO-SPY code

            self.det = Detector(device=self.device, mode="fusion", train_dataset=train_dataset, pretrain=True)
        finally:
            os.chdir(cwd)

    def score(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        probs = []
        for p in paths:
            x = self.det.model.test_transform(Image.open(p).convert("RGB")).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                v = self.det.model.predict(x)[0]
            probs.append(float(v.item() if hasattr(v, "item") else v))
        return logit(np.array(probs))


class MockDetector(BaseDetector):
    """Smoke tests only: a weak hand-made score (high-frequency energy) plus noise."""

    name = "mock"

    def __init__(self, seed: int = 0, noise: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def score(self, paths: list[str]) -> np.ndarray:
        import cv2

        vals = []
        for p in paths:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float64)
            hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
            vals.append(np.log(np.mean(hp**2) + 1e-6))
        v = np.array(vals)
        v = (v - v.mean()) / (v.std() + 1e-9)
        return v + self.noise * self.rng.normal(size=len(v))


DETECTORS = {"univfd": UnivFDDetector, "bfree": BFreeDetector, "cospy": CoSpyDetector, "mock": MockDetector}


def get_detector(name: str, cfg: dict[str, Any]) -> BaseDetector:
    if name not in DETECTORS:
        raise KeyError(f"Unknown detector {name}; choose from {list(DETECTORS)}")
    return DETECTORS[name](**cfg)
