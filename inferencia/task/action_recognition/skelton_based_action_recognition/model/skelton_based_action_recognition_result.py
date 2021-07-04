from dataclasses import dataclass
import numpy as np


@dataclass
class SkeltonBasedActionRecognitionResult:
    class_id: int
    class_name: str
    confidence: float
    confidences: np.ndarray
