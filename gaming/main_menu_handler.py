# from pydantic import BaseModel
# from typing import List
# from gaming.vlm_utils import analyze_game_info, action_in_options

# class Info(BaseModel):
#     menu_layout: str
#     menu_options: List[str]
#     selected_option: str

# def dummy_keyboard(action, action_repeat):
#     return f"{action} {action_repeat} times"

# def dummy_controller(cur_action_index, next_action_index, layout):
#     index_diff = cur_action_index - next_action_index
#     repeat_action_times = abs(index_diff)

#     if repeat_action_times == 0:
#         return dummy_keyboard("interact", 1)
#     elif layout == "horizontal":
#         action_direction = "move_left" if index_diff > 0 else "move_right"
#         return dummy_keyboard(action_direction, repeat_action_times)
#     elif layout == "vertical":
#         action_direction = "move_up" if index_diff > 0 else "move_down"        
#         return dummy_keyboard(action_direction, repeat_action_times)

# def main_menu_handler(llm, images):
#     """
#     Handles menu navigation logic using provided images (PIL list).
#     """
#     json_schema = Info.model_json_schema()
    
#     anal_prompt = ("You are a Gaming AI. Analyze the current state of the game. "
#                    "Provide a JSON of observations including spatial layout "
#                    "(vertical/horizontal/grid) and the list of menu options.")
    
#     # 1. Extract State
#     game_info = analyze_game_info(llm, anal_prompt, images, json_schema)
    
#     menu_options = game_info["menu_options"]
#     menu_layout = game_info["menu_layout"]
#     selected_option = game_info["selected_option"]

#     # 2. Decide Target
#     act_prompt = (f"You are a Gaming AI. Analyze the game.\n"
#                   f"Available options: {menu_options}\n\n"
#                   f"Choose the best option to proceed.")
    
#     next_action = action_in_options(llm, act_prompt, images, menu_options)
    
#     # 3. Calculate Controller Input
#     next_action_index = menu_options.index(next_action)
#     cur_selection_index = menu_options.index(selected_option)
    
#     control_command = dummy_controller(cur_selection_index, next_action_index, menu_layout)
#     print(f"ACTION DECIDED: {next_action}")
#     print(f"CONTROLLER SENDING: {control_command}")
    
#     return control_command



import time
from pydantic import BaseModel
from typing import List
# Assumed imports based on your prompt/previous context
from gaming.vlm_utils import analyze_game_info, action_in_options
from gaming.navigation_utils import calculate_menu_steps

class Info(BaseModel):
    menu_layout: str
    menu_options: List[str]
    selected_option: str

def execute_steps(controller, keys: List[str]):
    """Helper to push a list of keys to the controller thread."""
    for key in keys:
        print(f"Controller: Pressing {key}")
        controller.execute_action({
            "type": "key_press",
            "details": {"key": key, "hold_time": 0.05} # Short tap
        })
        # brief pause between inputs to ensure game registers them
        time.sleep(0.15) 

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
        
        # Get the list of 'up', 'down', 'z', etc.
        steps = calculate_menu_steps(cur_idx, target_idx, menu_layout)
        
        if steps:
            execute_steps(controller, steps)
        else:
            print(" No movement needed.")

    except ValueError as e:
        print(f" Error: Option mismatch in extraction vs decision. {e}")

    return target_option

if __name__ == "__main__":
    from interfaces.vllm_interface import JohnVLLM
    from interfaces.base import BaseModelConfig
    from PIL import Image

    images = [Image.open("debug_frames/img2.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]

    model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
        }
    model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
    llm = JohnVLLM(model_config).load_model(model_config)
    command=main_menu_handler(llm, images)
    print(command)