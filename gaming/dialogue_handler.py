from pydantic import BaseModel, Field
from typing import List, Optional
import time

# Shared Utilities
from gaming.vlm_utils import analyze_game_info, action_in_options
from gaming.navigation_utils import calculate_menu_steps, execute_steps, KEY_MAP

class Dialogue(BaseModel):
    is_dialogue_active: bool = Field(..., description="Is a dialogue box visible?")
    speaker_name: Optional[str] = Field(None, description="Name of speaker.")
    dialogue_text: Optional[str] = Field(None, description="Text content.")
    has_more_text_indicator: bool = Field(False, description="Indicator for more text.")
    player_choices: List[str] = Field(default_factory=list, description="List of choices.")
    selected_choice: Optional[str] = Field(None, description="Currently highlighted choice.")
    menu_layout: Optional[str] = Field(None, description="Layout (vertical/horizontal).")

def dialogue_handler(llm, images, controller):
    """
    Analyzes dialogue state. Handles both linear text advancement and choice selection.
    """
    json_schema = Dialogue.model_json_schema()

    anal_prompt = (
        "You are a Gaming AI playing a video game. Analyze the current state of the dialogue. "
        "Provide a JSON of your observations."
    )
    
    # 1. Analyze State
    print(" Analyzing Dialogue State...")
    game_info = analyze_game_info(llm, anal_prompt, images, json_schema)
    
    action_options = game_info.get("player_choices", [])
    current_selection = game_info.get("selected_choice")
    layout = game_info.get("menu_layout", "vertical")
    
    # 2. Decision Logic
    if action_options:
        print(f" Dialogue Choices Detected: {action_options}")
        
        # Safe Index Lookup
        try:
            cur_idx = action_options.index(current_selection) if current_selection in action_options else 0
        except ValueError:
            cur_idx = 0

        # Ask AI for choice
        act_prompt = (
            f"You are a Gaming AI. Dialogue choices: {action_options}\n"
            f"Choose the single best response."
        )
        target_action = action_in_options(llm, act_prompt, images, action_options)
        print(f" AI Decided: {target_action}")

        # Execute Movement + Confirm
        try:
            target_idx = action_options.index(target_action)
            steps = calculate_menu_steps(cur_idx, target_idx, layout)
            
            if steps:
                execute_steps(controller, steps)
                time.sleep(0.2) # Wait for cursor
            
            execute_steps(controller, KEY_MAP["confirm"])
            
        except ValueError:
            print(f" Error: Target '{target_action}' not in options.")

    else:
        # Linear Dialogue
        print(" Linear dialogue. Advancing...")
        execute_steps(controller, KEY_MAP["interact"])

    return game_info

if __name__ == "__main__":
    from interfaces.vllm_interface import JohnVLLM
    from interfaces.base import BaseModelConfig
    from PIL import Image
    from gaming.controls import InputControllerThread  # Actual controller import
    
    model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
        }
    model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
    llm = JohnVLLM(model_config).load_model(model_config)
    print(">>> Initializing Input Controller...")
    controller = InputControllerThread()
    controller.start()
    images = [Image.open("debug_frames/dialogue-20.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]

    dialogue_handler(llm, images,controller)
    controller.stop()
