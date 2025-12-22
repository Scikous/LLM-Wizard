import gc
import logging
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams, RequestOutputKind
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from Sagex.utils.special_chat_templates import get_special_template_for_model

import torch
# from model_utils import apply_chat_template # Assumed not needed if using tokenizer.apply_chat_template

from Sagex.interfaces.base import BaseModelConfig, JohnLLMAsyncBase, JohnLLMBase

class _VLLMInterfaceMixin:
    """
    A mixin to share prompt preparation and sampling parameter logic for vLLM implementations.
    """
    def _process_vision(self,
                        messages: List[Dict]
                        ):
        vision_data = {"multi_modal_data": {}, "mm_processor_kwargs": {}}
        
        # Determine patch size if available on processor, otherwise default
        patch_size = 14
        if hasattr(self.tokenizer, "image_processor") and hasattr(self.tokenizer.image_processor, "patch_size"):
             patch_size = self.tokenizer.image_processor.patch_size

        image_inputs, video_inputs, video_kwargs = self.vision_processor.process_inputs(messages, patch_size)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        vision_data.update({
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs
        })
        return vision_data
        
    def _update_chat_history(
        self,
        prompt: Optional[str],
        images: Optional[List],
        videos: Optional[List],
        assistant_prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        self.chat_history.add_user_message(prompt, images, videos)
        if assistant_prompt:
            self.chat_history.add_assistant_message(assistant_prompt)
        messages = self.chat_history.get_messages()
        return messages

    @staticmethod
    def _create_sampling_params(
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> SamplingParams:
        """
        Creates a vLLM SamplingParams object from a configuration dictionary.

        This helper function is responsible for parsing the generation configuration,
        specifically handling the dynamic creation of StructuredOutputsParams.

        Args:
            generation_config (Dict[str, Any], optional): A dictionary containing parameters
                for vLLM's SamplingParams, including a special 'guided_decoding' key.

        Returns:
            SamplingParams: A configured vLLM sampling parameters object.
        """
        config = generation_config.copy() if generation_config else {}

        guided_decoding_config = config.pop("guided_decoding", None)
        guided_decoding_params = None

        if guided_decoding_config:
            if isinstance(guided_decoding_config, dict) and len(guided_decoding_config) == 1:
                guided_decoding_params = StructuredOutputsParams(**guided_decoding_config)
            else:
                raise ValueError(
                    "'guided_decoding' in generation_config must be a dictionary "
                    "with a single key specifying the decoding type (e.g., 'json', 'regex')."
                )

        return SamplingParams(structured_outputs=guided_decoding_params, **config)


_DIALOGUE_GENERATOR_DOCSTRING = """
Generate text using the vLLM engine.

Args:
    prompt (str): The prompt to generate text from.
    assistant_prompt (str, optional): Optional prompt for the assistant to use as the base for its response.
    images (List, optional): List of image dictionaries for vision models.
    videos (List, optional): List of video dictionaries for vision models.
    generation_config (Dict[str, Any], optional): A dictionary containing all sampling and generation parameters.
    custom_wrapped (bool, optional): If True, treats prompt as if it's already formatted as role: role, content: content.
    Example for `generation_config` with guided JSON:
    generation_config = {{
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "guided_decoding": {{
            "json": '{{"type": "object", "properties": {{"name": {{"type": "string"}}}}}}'
        }}
    }}
Returns:
{return_type}: {return_description}
"""

# --- Asynchronous vLLM Implementation ---

class JohnVLLMAsync(JohnLLMAsyncBase, _VLLMInterfaceMixin):
    """Asynchronous implementation of JohnLLMBase using vLLM."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.engine: Optional[AsyncLLM] = None
        self.tokenizer: Optional[Any] = None
        self.current_request_id: Optional[str] = None

    @classmethod
    async def load_model(cls, config: BaseModelConfig) -> "JohnVLLMAsync":
        instance = cls(config)
        instance.logger.info(f"Initializing vLLM AsyncLLM for model: {config.model_path_or_id}")
        
        engine_args_dict = {
            "model": config.model_path_or_id,
        }
        engine_args_dict.update(config.model_init_kwargs)
        
        engine_args = AsyncEngineArgs(**engine_args_dict)
        instance.engine = AsyncLLM.from_engine_args(engine_args)
        instance.tokenizer = await instance.engine.get_tokenizer()
        
        if config.uses_special_chat_template:
            instance.tokenizer.chat_template = get_special_template_for_model(config.model_path_or_id)
            instance.logger.info(f"Special chat template loaded for model: {config.model_path_or_id}")

        return instance

    @staticmethod
    def _get_output_kind(output_kind_str: str) -> RequestOutputKind:
        if not output_kind_str or output_kind_str.upper() == "CUMULATIVE":
            return RequestOutputKind.CUMULATIVE
        if output_kind_str.upper() == "DELTA":
            return RequestOutputKind.DELTA
        return RequestOutputKind.CUMULATIVE

    async def dialogue_generator(self,
                            prompt: str,
                            assistant_prompt: Optional[str] = None,
                            images: Optional[List[Dict]] = None,
                            videos: Optional[List[Dict]] = None,
                            generation_config: Optional[Dict[str, Any]] = None,
                            custom_wrapped: Optional[bool] = False
                             ) -> AsyncGenerator[str, None]:
        
        if generation_config is None:
            generation_config = {}

        # Default to DELTA for async streaming if not specified
        if "output_kind" not in generation_config:
            generation_config["output_kind"] = "DELTA"
        generation_config["output_kind"] = self._get_output_kind(generation_config["output_kind"])

        # 1. Update Conversation / History
        if not custom_wrapped:
            messages = self._update_chat_history(prompt, images, videos, assistant_prompt)
        else:
            messages = prompt # Assumes prompt is list of dicts if custom_wrapped, or raw string
        
        # 2. Apply Chat Template
        add_generation_prompt = True
        continue_final_message = False
        if assistant_prompt:
            add_generation_prompt = False
            continue_final_message = True
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt, 
            continue_final_message=continue_final_message
        )

        # 3. Vision Processing        
        if self.vision_processor and (images or videos):
            vision_data = self._process_vision(messages)
            prompt = {"prompt": prompt, "multi_modal_data": vision_data["multi_modal_data"], "mm_processor_kwargs": vision_data["mm_processor_kwargs"]}

        # 4. Generate
        sampling_params = self._create_sampling_params(generation_config)
        self.current_request_id = f"john-llm-{uuid.uuid4().hex}"
        
        results_generator = self.engine.generate(
            prompt, 
            sampling_params=sampling_params, 
            request_id=self.current_request_id
        )
        
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            yield current_text
            
        self.current_request_id = None
    # Set the docstring dynamically
    dialogue_generator.__doc__ = _DIALOGUE_GENERATOR_DOCSTRING.format(
        return_type="str",
        return_description="The complete generated text as a single string."
    )
    async def cancel_dialogue_generation(self):
        if self.current_request_id:
            try:
                await self.engine.abort(self.current_request_id)
                self.logger.info(f"Aborted vLLM request: {self.current_request_id}")
            except Exception as e:
                self.logger.warning(f"Failed to abort vLLM request: {e}")
            self.current_request_id = None

    async def warmup(self):
        self.logger.info("Warming up the async vLLM engine...")
        try:
            gen_config = {"max_tokens": 10}
            async for _ in self.dialogue_generator(prompt="Hello", generation_config=gen_config):
                pass
            self.logger.info("Async vLLM engine warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during async vLLM warmup: {e}")

    async def cleanup(self):
        if self.current_request_id:
            await self.cancel_dialogue_generation()
        if self.engine:
            del self.engine
        self.engine = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up async vLLM resources.")

# --- Synchronous vLLM Implementation ---
class JohnVLLM(JohnLLMBase, _VLLMInterfaceMixin):
    """Synchronous implementation of JohnLLMBase using vLLM's LLM."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.engine: Optional[LLM] = None
        self.tokenizer: Optional[Any] = None
        # self.vision_processor and self.chat_history are initialized in super()

    @classmethod
    def load_model(cls, config: BaseModelConfig) -> "JohnVLLM":
        instance = cls(config)
        instance.logger.info(f"Initializing vLLM LLM for model: {config.model_path_or_id}")
        instance.engine = LLM(
            model=config.model_path_or_id,
            **config.model_init_kwargs
        )
        
        # Check if vision processor exists (set by Base based on config or default)
        if instance.vision_processor:
            from transformers import AutoProcessor
            instance.tokenizer = AutoProcessor.from_pretrained(config.model_path_or_id)
        else:
            instance.tokenizer = instance.engine.get_tokenizer()

        if config.uses_special_chat_template:
            instance.tokenizer.chat_template = get_special_template_for_model(config.model_path_or_id)
            instance.logger.info(f"Special chat template loaded for model: {config.model_path_or_id}")
            
        return instance

    def dialogue_generator(self,
                       prompt: str,
                       assistant_prompt: Optional[str] = None,
                       images: Optional[List[Dict]] = None,
                       videos: Optional[List[Dict]] = None,
                       generation_config: Optional[Dict[str, Any]] = None,
                       custom_wrapped: Optional[bool] = False
                       ) -> str:
        
        
        # 1. Update Conversation / History
        if not custom_wrapped:
            messages = self._update_chat_history(prompt, images, videos, assistant_prompt)
        else:
            messages = prompt # Assumes prompt is list of dicts if custom_wrapped, or raw string
        
        # 2. Apply Chat Template
        add_generation_prompt = True
        continue_final_message = False
        if assistant_prompt:
            add_generation_prompt = False
            continue_final_message = True
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt, 
            continue_final_message=continue_final_message
        )

        # 3. Vision Processing        
        if self.vision_processor and (images or videos):
            vision_data = self._process_vision(messages)
            prompt = {"prompt": prompt, "multi_modal_data": vision_data["multi_modal_data"], "mm_processor_kwargs": vision_data["mm_processor_kwargs"]}

        # 4. Generate
        sampling_params = self._create_sampling_params(generation_config)
        print(f">>>Prompt: {prompt}\nSampling_params: {sampling_params}\n\n")
        outputs = self.engine.generate(prompt, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # Set the docstring dynamically
    dialogue_generator.__doc__ = _DIALOGUE_GENERATOR_DOCSTRING.format(
        return_type="str",
        return_description="The complete generated text as a single string."
    )

    def cancel_dialogue_generation(self):
        self.logger.warning("Synchronous vLLM does not support cancellation of a running generator.")

    def warmup(self):
        self.logger.info("Warming up the sync vLLM engine...")
        try:
            # Consume the generator to ensure execution
            self.dialogue_generator(prompt="Hello")
            self.logger.info("Sync vLLM engine warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during sync vLLM warmup: {e}")

    def cleanup(self):
        del self.engine
        self.engine = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up sync vLLM resources.")