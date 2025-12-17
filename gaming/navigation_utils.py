from typing import List, Literal

# Configuration for default keys
KEY_MAP = {
    "interact": "z",  # Standard RPG Maker/Paper Lily interact key
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right"
}

def calculate_menu_steps(
    current_idx: int, 
    target_idx: int, 
    layout: Literal["vertical", "horizontal"]
) -> List[str]:
    """
    Calculates the sequence of keys needed to move from current_idx to target_idx.
    Returns a list of keys (e.g., ['up', 'up'] or ['z']).
    """
    steps = []
    
    # Calculate difference (Current - Target)
    # UI List Logic: Index 0 is Top/Left. Index 1 is Bottom/Right.
    # Diff > 0: We are "below" or "right of" target. Move Up/Left.
    # Diff < 0: We are "above" or "left of" target. Move Down/Right.
    index_diff = current_idx - target_idx
    repeats = abs(index_diff)

    # If we are already at the target, the action is to Interact
    if repeats == 0:
        return [KEY_MAP["interact"]]

    direction_key = ""
    
    if layout == "horizontal":
        if index_diff > 0:
            direction_key = KEY_MAP["left"]
        else:
            direction_key = KEY_MAP["right"]
            
    elif layout == "vertical":
        if index_diff > 0:
            direction_key = KEY_MAP["up"]
        else:
            direction_key = KEY_MAP["down"]

    # Add the movement keys to the list n times
    steps.extend([direction_key] * repeats)
    
    # Optional: If you want to move AND select immediately, uncomment the next line.
    # For now, we only move to the target. The next iteration will trigger the interact.
    # steps.append(KEY_MAP["interact"]) 
    
    return steps