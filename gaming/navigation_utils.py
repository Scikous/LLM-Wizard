from typing import List, Literal, Union

# Configuration for default keys
KEY_MAP = {
    "interact": "z",
    "confirm": "z",
    "back": "x",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right"
}

def execute_steps(controller, keys: Union[List[str], str], duration: float = 0.05, stagger: float = 0.1):
    """
    Universal function to execute key presses using the advanced InputControllerThread.
    
    Optimizations:
    - Accepts single string or list of keys.
    - Uses the controller's internal 'stagger_delay' to handle sequences naturally 
      without manual sleep loops in the main thread.
    - Relies on the controller's priority queue to handle KeyUp/KeyDown conflicts 
      during rapid inputs.
    """
    if not keys:
        return

    # Normalize to list
    if isinstance(keys, str):
        keys = [keys]

    # Send single batched command
    # The controller's _handle_key_press will build the timeline 
    # and execute them with the specific stagger.
    controller.execute_action({
        "type": "key_press",
        "details": {
            "key": keys,
            "hold_time": duration,
            "stagger_delay": stagger
        }
    })

def ensure_menu_selection(controller):
    """
    Performs a 'Wake Up' sequence to guarantee a UI element is highlighted.
    
    Why: Some games don't highlight the first option by default until an input is received.
    Sequence: Down -> Up -> Right -> Left
    """
    wake_up_sequence = [
        KEY_MAP['down'], 
        KEY_MAP['up'], 
        KEY_MAP['right'], 
        KEY_MAP['left']
    ]
    
    # We send this as a quick ripple of inputs.
    # Stagger is slightly longer to ensure the UI animation has time to react 
    # so the visual cursor appears.
    print("Navigation: Executing Menu Wake-up Sequence...")
    execute_steps(controller, wake_up_sequence, duration=0.05, stagger=0.10)

def calculate_menu_steps(
    current_idx: int, 
    target_idx: int, 
    layout: Literal["vertical", "horizontal"]
) -> List[str]:
    """
    Calculates the sequence of keys needed to move from current_idx to target_idx.
    """
    index_diff = current_idx - target_idx
    repeats = abs(index_diff)

    if repeats == 0:
        return []

    direction_key = ""
    
    if layout == "horizontal":
        direction_key = KEY_MAP["left"] if index_diff > 0 else KEY_MAP["right"]
    elif layout == "vertical":
        direction_key = KEY_MAP["up"] if index_diff > 0 else KEY_MAP["down"]

    return [direction_key] * repeats