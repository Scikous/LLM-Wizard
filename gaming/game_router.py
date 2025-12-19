from pydantic import BaseModel, Field
from typing import Literal
from gaming.vlm_utils import analyze_game_info

class GameState(BaseModel):
    """
    Classifies the current visual state of the video game.
    """
    current_state: Literal["main_menu", "dialogue", "gameplay", "unknown"] = Field(
        ...,
        description=(
            "The primary mode of the game. "
            "'main_menu': Title screens, pause menus, inventory screens, or list-based UI. "
            "'dialogue': Text boxes at the bottom, visual novel style conversations, or choice prompts. "
            "'gameplay': Standard exploration, walking, combat, or cutscenes with no UI overlay."
        )
    )
    confidence_score: float = Field(
        ..., 
        description="Confidence in this classification (0.0 to 1.0)."
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why this state was chosen (e.g. 'Text box visible', 'Start button visible')."
    )

def decide_game_state(llm, images):
    """
    Takes a screenshot and determines which handler should run.
    """
    json_schema = GameState.model_json_schema()
    
    prompt = (
        "You are a Gaming AI. Analyze the screenshot and classify the current game state. "
        "Is the player in a Menu, in a Dialogue?"
    )
    
    # We use the shared VLM utility to get the JSON
    state_info = analyze_game_info(llm, prompt, images, json_schema)
    
    # Return the parsed Pydantic object (or just the dict)
    return state_info