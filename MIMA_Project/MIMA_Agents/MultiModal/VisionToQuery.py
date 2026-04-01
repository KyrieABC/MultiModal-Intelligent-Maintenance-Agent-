from __future__ import annotations

from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer, util

FAULT_LABELS = [
    "oil leak",
    "coolant leak",
    "corrosion",
    "burn mark",
    "belt wear",
    "bearing damage",
    "overheating",
    "loose wiring",
    "cracked housing",
    "rusted fastener",
    "misalignment",
    "frayed cable",
    "hydraulic residue",
    "dust buildup",
    "fan obstruction",
    "seal degradation",
]


class VisionToQuery:
    """Map an equipment photo to semantic maintenance labels using CLIP embeddings."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.label_embeddings = self.model.encode(FAULT_LABELS, convert_to_tensor=True)

    def extract_labels(self, image_path: str, top_k: int = 5) -> list[str]:
        image = Image.open(Path(image_path)).convert("RGB")
        image_embedding = self.model.encode(image, convert_to_tensor=True)
        similarities = util.cos_sim(image_embedding, self.label_embeddings)[0]
        ranked = sorted(
            zip(FAULT_LABELS, similarities.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [label for label, _ in ranked[:top_k]]

    def build_semantic_query(self, user_question: str, image_path: str | None) -> tuple[str, list[str]]:
        if not image_path:
            return user_question, []
        labels = self.extract_labels(image_path=image_path)
        vision_context = ", ".join(labels)
        query = (
            f"{user_question}. Visual inspection signals: {vision_context}. "
            "Find troubleshooting procedures, root causes, safety steps, and repair guidance."
        )
        return query, labels
