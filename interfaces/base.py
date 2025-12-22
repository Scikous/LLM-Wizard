#  interfaces/base.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, AsyncGenerator, Union

from Sagex.processors.vision import get_vision_processor, BaseVisionProcessor
from Sagex.chat_history import ChatHistory

# --- Base Configuration ---
@dataclass
class BaseModelConfig:
    """
    Unified configuration for LLM models.
    Backend-specific settings (like gpu_memory_utilization for vLLM) 
    should go into model_init_kwargs.
    """
    model_path_or_id: str
    uses_special_chat_template: bool = False
    # Path to tokenizer if different from model (common in ExLlama)
    tokenizer_path_or_id: Optional[str] = None 
    character_name: str = 'assistant'
    instructions: str = ""
    
    # Optional instances or overrides
    chat_history: Optional[ChatHistory] = None
    vision_processor: Optional[BaseVisionProcessor] = None
    
    # Backend specific arguments (e.g. vllm's gpu_memory_utilization)
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)


# --- Synchronous Base Class ---
class JohnLLMBase(ABC):
    """Abstract base class for synchronous John LLM models."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.character_name = config.character_name
        self.instructions = config.instructions

        # Initialize ChatHistory
        if self.config.chat_history:
            self.chat_history = self.config.chat_history
        else:
            self.chat_history = ChatHistory(system_instructions=self.instructions)

        # Initialize Vision Processor
        if self.config.vision_processor:
            self.vision_processor = self.config.vision_processor
        else:
            # Default fetch based on model ID
            self.vision_processor = get_vision_processor(self.config.model_path_or_id)

    @classmethod
    @abstractmethod
    def load_model(cls, config: BaseModelConfig) -> "JohnLLMBase":
        """Loads all necessary model resources and returns an instance of the class."""
        pass

    @abstractmethod
    def warmup(self):
        """Performs any necessary warmup operations."""
        pass

    @abstractmethod
    def dialogue_generator(
        self,
        prompt: str,
        assistant_prompt: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        videos: Optional[List[Dict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        custom_wrapped: Optional[bool] = False
    ) -> Generator[str, None, None]:
        """Generates a response to a prompt, yielding tokens as they are generated."""
        pass

    @abstractmethod
    def cancel_dialogue_generation(self):
        """Requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleans up all resources held by the model."""
        pass


class JohnLLMAsyncBase(ABC):
    """Abstract base class for asynchronous John LLM models."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.character_name = config.character_name
        self.instructions = config.instructions

        # Initialize ChatHistory
        if self.config.chat_history:
            self.chat_history = self.config.chat_history
        else:
            self.chat_history = ChatHistory(system_instructions=self.instructions)

        # Initialize Vision Processor
        if self.config.vision_processor:
            self.vision_processor = self.config.vision_processor
        else:
            self.vision_processor = get_vision_processor(self.config.model_path_or_id)

    @classmethod
    @abstractmethod
    async def load_model(cls, config: BaseModelConfig) -> "JohnLLMAsyncBase":
        """Asynchronously loads all necessary model resources."""
        pass

    @abstractmethod
    async def warmup(self):
        """Asynchronously performs any necessary warmup operations."""
        pass

    @abstractmethod
    async def dialogue_generator(
        self, 
        prompt: str, 
        assistant_prompt: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        videos: Optional[List[Dict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        custom_wrapped: Optional[bool] = False
    ) -> AsyncGenerator[str, None]:
        """
        Generates text and yields token deltas (strings).
        Must accept `generation_config` for parameters like max_tokens, temperature, etc.
        """
        pass

    @abstractmethod
    async def cancel_dialogue_generation(self):
        """Asynchronously requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Asynchronously cleans up all resources held by the model."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False