from pydantic import BaseModel, Field
from typing import List, Tuple

class GameEntity(BaseModel):
    label: str = Field(..., description="Name of the object (e.g., 'player_character', 'door', 'enemy', 'rock').")
    # VLM usually outputs [x1, y1, x2, y2] in 1000x1000 coordinate space
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] coordinates (0-1000 scale).")

class GameplayScan(BaseModel):
    """
    Structured analysis of the gameplay screen for navigation.
    """
    character: GameEntity = Field(..., description="The player's character.")
    target: GameEntity = Field(..., description="The objective to move towards (e.g., a door, an NPC, a chest).")
    obstacles: List[GameEntity] = Field(default_factory=list, description="List of static objects that block movement.")
    
    # Helper to convert VLM [x1, y1, x2, y2] to OpenCV [x, y, w, h]
    def get_cv2_rect(self, entity: GameEntity) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = entity.bbox
        return (x1, y1, x2 - x1, y2 - y1)