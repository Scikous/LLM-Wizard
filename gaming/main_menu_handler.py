import time
from pydantic import BaseModel
from typing import List
# Assumed imports based on your prompt/previous context
from gaming.vlm_utils import analyze_game_info, action_in_options
from gaming.navigation_utils import calculate_menu_steps, execute_steps, KEY_MAP


class Info(BaseModel):
    menu_layout: str
    menu_options: List[str]
    selected_option: str

def main_menu_handler(llm, images, controller):
    """
    Analyzes menu, decides action, and physically inputs the keys.
    """
    json_schema = Info.model_json_schema()
    
    anal_prompt = ("You are a Gaming AI. Analyze the current state of the game. "
                   "Provide a JSON of observations including spatial layout "
                   "(vertical/horizontal) and the list of menu options.")
    
    # 1. Extract State
    print(" Analyzing Menu State...")
    game_info = analyze_game_info(llm, anal_prompt, images, json_schema)
    
    menu_options = game_info["menu_options"]
    menu_layout = game_info["menu_layout"]
    selected_option = game_info["selected_option"]
    
    print(f" Layout: {menu_layout} | Selected: {selected_option}")

    # 2. Decide Target
    act_prompt = (f"You are a Gaming AI. Analyze the game.\n"
                  f"Available options: {menu_options}\n\n"
                  f"Choose the single best option to proceed.")
    
    target_option = action_in_options(llm, act_prompt, images, menu_options)
    print(f" AI Target: {target_option}")

    # 3. Calculate & Execute Physical Actions
    try:
        cur_idx = menu_options.index(selected_option)
        target_idx = menu_options.index(target_option)
        
        # Get movement keys
        steps = calculate_menu_steps(cur_idx, target_idx, menu_layout)
        
        if steps:
            print(f" Controller: Moving {steps}")
            # Use new batch executor
            execute_steps(controller, steps)
            # Small buffer for UI animation to settle before confirming
            time.sleep(0.2)
        
        # Confirm Selection
        print(" Controller: Confirming...")
        execute_steps(controller, KEY_MAP["confirm"])

    except ValueError as e:
        print(f" Error: Option mismatch in extraction vs decision. {e}")

    return target_option

if __name__ == "__main__":
    from interfaces.vllm_interface import JohnVLLM
    from interfaces.base import BaseModelConfig
    from gaming.controls import InputControllerThread  # Actual controller import
    from PIL import Image

    images = [Image.open("debug_frames/img2.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]

    model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
        }
    model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
    llm = JohnVLLM(model_config).load_model(model_config)
    print(">>> Initializing Input Controller...")
    controller = InputControllerThread()
    controller.start()
    command=main_menu_handler(llm, images, controller)
    print(command)
    controller.stop()
