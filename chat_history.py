# # Sagex/conversation.py
# from typing import List, Dict, Optional, Union, Any

# class Conversation:
#     def __init__(self, system_instructions: str = ""):
#         self.messages: List[Dict[str, Any]] = []
#         if system_instructions:
#             self.messages.append({"role": "system", "content": system_instructions})

#     def add_custom_message(self, message):
#         self.messages.append(message)

#     def add_user_message(self, text: str, images: Optional[List] = None, videos: Optional[List] = None):
#         """
#         Standard helper for the most common case: [Images -> Videos -> Text].
#         """
#         # If simple text only, keep it simple
#         if not images and not videos:
#             self.messages.append({"role": "user", "content": text})
#             return

#         # Build content list
#         content = []
#         if images:
#             # We assume simple placeholders for the 'Basic' case
#             content.extend([{"type": "image", "image": img} for img in images])
#         if videos:
#             content.extend([{"type": "video", "video": vid} for vid in videos])
        
#         content.append({"type": "text", "text": text})
        
#         self.messages.append({"role": "user", "content": content})

#     def add_assistant_message(self, text: str):
#         self.messages.append({"role": "assistant", "content": text})

#     def get_messages(self) -> List[Dict[str, Any]]:
#         """Returns the list of messages."""
#         return self.messages

#     def clear(self, keep_system_instructions: bool = True):
#         # Logic to reset but keep system prompt if desired
#         if keep_system_instructions and self.messages and self.messages[0]['role'] == 'system':
#             self.messages = self.messages[0]
#         else:
#             self.messages = []


# Sagex/chat_history.py
from typing import List, Dict, Optional, Any

class ChatHistory:
    def __init__(self, system_instructions: str = "", max_messages: Optional[int] = None):
        """
        Args:
            system_instructions: The system prompt.
            max_messages: Maximum number of messages to keep in history (excluding system prompt).
                          If None, history is unlimited.
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        
        if system_instructions:
            self.messages.append({"role": "system", "content": system_instructions})

    def _enforce_limit(self):
        """
        Trims the message history to enforce max_messages limit.
        Always preserves the system prompt if it exists at index 0.
        """
        if self.max_messages is None:
            return

        has_system = len(self.messages) > 0 and self.messages[0].get("role") == "system"
        effective_history = self.messages[1:] if has_system else self.messages
        
        if len(effective_history) > self.max_messages:
            # Slice to keep the most recent N messages
            effective_history = effective_history[-self.max_messages:]
            
            # Reconstruct messages list
            self.messages = [self.messages[0]] + effective_history if has_system else effective_history

    def add_custom_message(self, message):
        self.messages.append(message)
        self._enforce_limit()

    def add_user_message(self, text: str, images: Optional[List] = None, videos: Optional[List] = None):
        """
        Standard helper for the most common case: [Images -> Videos -> Text].
        """
        # If simple text only, keep it simple
        if not images and not videos:
            self.messages.append({"role": "user", "content": text})
            self._enforce_limit()
            return

        # Build content list
        content = []
        if images:
            content.extend([{"type": "image", "image": img} for img in images])
        if videos:
            content.extend([{"type": "video", "video": vid} for vid in videos])
        
        content.append({"type": "text", "text": text})
        
        self.messages.append({"role": "user", "content": content})
        self._enforce_limit()

    def add_assistant_message(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._enforce_limit()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the list of messages."""
        return self.messages

    def clear(self, keep_system_instructions: bool = True):
        # Logic to reset but keep system prompt if desired
        if keep_system_instructions and self.messages and self.messages[0]['role'] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []