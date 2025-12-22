from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

class BaseVisionProcessor(ABC):
    @abstractmethod
    def process_inputs(self, messages: List[Dict], tokenizer: Any) -> Dict[str, Any]:
        """Returns the dictionary to be merged into vLLM inputs."""
        pass

class QwenVisionProcessor(BaseVisionProcessor):
    def process_inputs(self, messages: List[Dict], image_patch_size: Optional[int] = 14) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info
        # Qwen specific logic
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        return image_inputs, video_inputs, video_kwargs

class StandardVisionProcessor(BaseVisionProcessor):
    def process_inputs(self, messages: List[Dict], tokenizer: Any) -> Dict[str, Any]:
        # Logic for standard LLaVA or other models if needed
        # For vLLM, often just passing the image in multi_modal_data is enough
        # This allows future expansion without touching the main class
        return {} 

def get_vision_processor(model_type: str) -> BaseVisionProcessor:
    if "qwen" in model_type.lower():
        return QwenVisionProcessor()
    return StandardVisionProcessor()