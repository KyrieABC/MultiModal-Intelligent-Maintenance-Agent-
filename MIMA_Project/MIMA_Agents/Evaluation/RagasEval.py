from __future__ import annotations

import pandas as pd

from ragas import EvaluationDataset, evaluate
from ragas.metrics import _faithfulness, _context_precision

def run_ragas_evaluation(samples: list[dict]) -> pd.DataFrame:
    """
    Expected sample format:
    {
        "user_input": "question",
        "response": "model answer",
        "retrieved_contexts": ["chunk1", "chunk2"],
        "reference": "gold answer"
    }
    """
    # Convert python list into RAGAS-Compatible dataset
    # EvaluationDataset is structure expected by evalute()
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        # _faithfulness: Is the model answer grounded in retrieved context
        # _context_precision: How relevant are the retrieved chunks to the question
        metrics=[_faithfulness,_context_precision],      
    )
    return result.to_pandas