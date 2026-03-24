"""
Real survey data loader: ingest CSV/JSON, compute reference distributions,
and produce segmented distributions for calibration.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.option_space import canonicalize_distribution, canonicalize_option
from core.rng import make_rng_pack

logger = logging.getLogger(__name__)


@dataclass
class RealSurveyData:
    """Loaded real-world survey data for a single question."""

    question: str
    responses: List[str]
    demographics: List[Dict[str, str]] = field(default_factory=list)
    question_model_key: str = "food_delivery_frequency"

    @property
    def n_responses(self) -> int:
        return len(self.responses)

    def to_reference_distribution(self) -> Dict[str, float]:
        """Aggregate into a probability distribution."""
        if not self.responses:
            return {}
        counts: Dict[str, int] = {}
        for r in self.responses:
            cr = canonicalize_option(self.question_model_key, r)
            counts[cr] = counts.get(cr, 0) + 1
        total = sum(counts.values())
        return canonicalize_distribution(
            self.question_model_key,
            {k: round(v / total, 4) for k, v in counts.items()},
        )

    def to_segmented_distributions(
        self, segment_by: str
    ) -> Dict[str, Dict[str, float]]:
        """Split by a demographic column and compute per-segment distributions."""
        segments: Dict[str, List[str]] = {}
        for i, resp in enumerate(self.responses):
            demo = self.demographics[i] if i < len(self.demographics) else {}
            seg_val = demo.get(segment_by, "unknown")
            segments.setdefault(seg_val, []).append(resp)

        result: Dict[str, Dict[str, float]] = {}
        for seg, resps in segments.items():
            counts: Dict[str, int] = {}
            for r in resps:
                counts[r] = counts.get(r, 0) + 1
            total = sum(counts.values())
            result[seg] = {k: round(v / total, 4) for k, v in counts.items()}
        return result

    def holdout_split(
        self, train_fraction: float = 0.8, seed: int = 42
    ) -> tuple["RealSurveyData", "RealSurveyData"]:
        """Split into train/test sets."""
        rng = make_rng_pack(f"calibration_holdout:{self.question}", base_seed=seed).np_rng
        n = len(self.responses)
        indices = rng.permutation(n)
        split = int(n * train_fraction)

        train_idx = indices[:split]
        test_idx = indices[split:]

        def _subset(idx):
            return RealSurveyData(
                question=self.question,
                responses=[self.responses[i] for i in idx],
                demographics=[self.demographics[i] for i in idx]
                if self.demographics else [],
                question_model_key=self.question_model_key,
            )

        return _subset(train_idx), _subset(test_idx)

    @classmethod
    def from_raw(
        cls,
        question: str,
        responses: List[str],
        demographics: Optional[List[Dict[str, str]]] = None,
    ) -> RealSurveyData:
        return cls(
            question=question,
            responses=responses,
            demographics=demographics or [],
        )


class RealDataLoader:
    """Load real survey data from CSV or JSON files."""

    def load_csv(
        self,
        path: str,
        question_col: str,
        answer_col: str,
        demographics_cols: Optional[List[str]] = None,
    ) -> Dict[str, RealSurveyData]:
        """Load CSV, grouping by question column. Returns question -> RealSurveyData."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        reader = csv.DictReader(StringIO(text))

        grouped: Dict[str, tuple] = {}
        for row in reader:
            q = row.get(question_col, "")
            ans = row.get(answer_col, "")
            if not q or not ans:
                continue
            if q not in grouped:
                grouped[q] = ([], [])
            grouped[q][0].append(ans)
            demo = {}
            if demographics_cols:
                for col in demographics_cols:
                    demo[col] = row.get(col, "")
            grouped[q][1].append(demo)

        return {
            q: RealSurveyData(question=q, responses=resps, demographics=demos)
            for q, (resps, demos) in grouped.items()
        }

    def load_json(self, path: str) -> Dict[str, RealSurveyData]:
        """Load JSON with structure: {question: {responses: [...], demographics: [...]}}"""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))

        result: Dict[str, RealSurveyData] = {}
        if isinstance(data, dict):
            for q, qdata in data.items():
                if isinstance(qdata, dict):
                    result[q] = RealSurveyData(
                        question=q,
                        responses=qdata.get("responses", []),
                        demographics=qdata.get("demographics", []),
                    )
                elif isinstance(qdata, list):
                    result[q] = RealSurveyData(question=q, responses=qdata)
        return result

    def load_csv_text(
        self,
        text: str,
        question_col: str,
        answer_col: str,
        demographics_cols: Optional[List[str]] = None,
    ) -> Dict[str, RealSurveyData]:
        """Load CSV from text content."""
        reader = csv.DictReader(StringIO(text))
        grouped: Dict[str, tuple] = {}
        for row in reader:
            q = row.get(question_col, "")
            ans = row.get(answer_col, "")
            if not q or not ans:
                continue
            if q not in grouped:
                grouped[q] = ([], [])
            grouped[q][0].append(ans)
            demo = {}
            if demographics_cols:
                for col in demographics_cols:
                    demo[col] = row.get(col, "")
            grouped[q][1].append(demo)

        return {
            q: RealSurveyData(question=q, responses=resps, demographics=demos)
            for q, (resps, demos) in grouped.items()
        }
