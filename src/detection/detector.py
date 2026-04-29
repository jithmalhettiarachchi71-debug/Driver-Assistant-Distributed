"""
YOLO Object Detector using ONNX Runtime.

Provides YOLODetector class for running inference on frames
with frame skipping and result caching for performance.
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .preprocessing import preprocess_for_yolo, scale_boxes
from .postprocessing import process_yolo_output, create_detections, create_detections_batched
from .result import Detection, DetectionResult


class YOLODetector:
    """
    YOLO Object Detector using ONNX Runtime.
    
    Features:
    - Loads ONNX model with CPU execution provider
    - Frame skipping for performance (configurable)
    - Result caching between detections
    - Confidence and IoU thresholds
    
    Usage:
        detector = YOLODetector("models/object.onnx")
        result = detector.detect(frame)
        for det in result.detections:
            print(f"{det.label.value}: {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        frame_skip: int = 2,
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to ONNX model file
            conf_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            input_size: Model input size (square)
            frame_skip: Process every N frames (1=all, 2=half, etc.)
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required for detection. "
                "Install with: pip install onnxruntime"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.frame_skip = max(1, frame_skip)
        
        # Load ONNX model
        self._session = self._load_model()
        self._input_name = self._session.get_inputs()[0].name
        
        # Frame counting and caching
        self._frame_count = 0
        self._cached_result: Optional[DetectionResult] = None
        
        # Performance tracking
        self._last_inference_time = 0.0
    
    def _load_model(self) -> "ort.InferenceSession":
        """Load ONNX model with CPU execution provider."""
        # Use only CPU provider for Raspberry Pi compatibility
        providers = ["CPUExecutionProvider"]
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use single thread for deterministic performance on Pi
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        
        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )
        
        return session
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a frame.
        
        Uses frame skipping - only runs inference every N frames,
        returning cached results for skipped frames.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            
        Returns:
            DetectionResult with list of Detection objects
        """
        self._frame_count += 1
        timestamp = time.monotonic()
        
        # Check if we should skip this frame
        if self._should_skip_frame():
            # Return cached result with updated timestamp
            if self._cached_result is not None:
                return DetectionResult(
                    detections=self._cached_result.detections,
                    timestamp=timestamp,
                    latency_ms=0.0,
                    from_cache=True,
                )
            # No cache yet, must run inference
        
        # Run inference
        start_time = time.perf_counter()
        
        # Preprocess
        input_tensor, scale, pad = preprocess_for_yolo(
            frame, 
            target_size=(self.input_size, self.input_size)
        )
        
        # Run model
        outputs = self._session.run(None, {self._input_name: input_tensor})
        output = outputs[0]  # Shape: (1, 84, 8400)
        
        # Post-process
        raw_detections = process_yolo_output(
            output,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            filter_relevant_only=True,
        )
        
        # OPT-1: Batch scale all boxes at once instead of per-detection loop
        h, w = frame.shape[:2]
        if raw_detections:
            class_ids = [d[0] for d in raw_detections]
            confidences = [d[1] for d in raw_detections]
            boxes = np.array([d[2] for d in raw_detections])
            
            # Batch scale all boxes in single operation
            scaled_boxes = scale_boxes(boxes, scale, pad, (w, h))
            
            # Create detections from batched results
            detections = create_detections_batched(
                class_ids, confidences, scaled_boxes, timestamp
            )
        else:
            detections = []
        
        # Calculate inference time
        inference_time = time.perf_counter() - start_time
        self._last_inference_time = inference_time
        
        # Create and cache result (latency_ms = seconds * 1000)
        result = DetectionResult(
            detections=detections,
            timestamp=timestamp,
            latency_ms=inference_time * 1000,
            from_cache=False,
        )
        self._cached_result = result
        
        return result
    
    def _should_skip_frame(self) -> bool:
        """Check if current frame should be skipped."""
        if self.frame_skip <= 1:
            return False
        return (self._frame_count % self.frame_skip) != 1
    
    def reset(self) -> None:
        """Reset frame counter and cache."""
        self._frame_count = 0
        self._cached_result = None
    
    @property
    def last_inference_time(self) -> float:
        """Get the last inference time in seconds."""
        return self._last_inference_time
    
    @property
    def model_info(self) -> dict:
        """Get model information."""
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        
        return {
            "model_path": str(self.model_path),
            "input_name": inputs[0].name,
            "input_shape": inputs[0].shape,
            "output_name": outputs[0].name,
            "output_shape": outputs[0].shape,
            "providers": self._session.get_providers(),
        }
    
    def __repr__(self) -> str:
        return (
            f"YOLODetector(model={self.model_path.name}, "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, "
            f"skip={self.frame_skip})"
        )
