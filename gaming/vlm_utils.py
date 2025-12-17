import json
from PIL import Image

# Shared assistant prefix to nudge the model toward JSON
ASSISTANT_PREFIX = "{\n"

def analyze_game_info(llm, prompt, images, schema):
    """Analyzes a game state and returns structured JSON."""
    guided_json_config = {
        "max_tokens": 1028,
        "temperature": 0.2,
        "skip_special_tokens": False,
        "guided_decoding": {
            "json": schema
        }
    }

    resp = llm.dialogue_generator(
        prompt=prompt, 
        assistant_prompt=ASSISTANT_PREFIX, 
        images=images, 
        generation_config=guided_json_config, 
        add_generation_prompt=False, 
        continue_final_message=True
    )
    return json.loads(resp)

def action_in_options(llm, prompt, images, options):
    """Forces the VLM to select one option from a provided list using regex."""
    act_regex = "|".join(options)
    guided_regex_config = {
        "max_tokens": 1028,
        "temperature": 0.2,
        "skip_special_tokens": False,
        "guided_decoding": {
            "regex": act_regex
        }
    }

    return llm.dialogue_generator(
        prompt=prompt, 
        assistant_prompt=ASSISTANT_PREFIX, 
        images=images, 
        generation_config=guided_regex_config, 
        add_generation_prompt=False, 
        continue_final_message=True
    )