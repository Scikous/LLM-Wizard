import cv2
import time
import traceback
from PIL import Image

# Core Interfaces
from interfaces.vllm_interface import JohnVLLM
from interfaces.base import BaseModelConfig
from gaming.controls import InputControllerThread

# Local Modules
from gaming.game_capture import CaptureManager, SystemConfig
from gaming.game_router import decide_game_state
from gaming.main_menu_handler import main_menu_handler
from gaming.dialogue_handler import dialogue_handler

def capture_latest_frame(capture_manager):
    """Helper to grab the freshest frame from the capture thread."""
    capture_manager.start_capture()
    time.sleep(0.1) # Short buffer fill
    raw_frames = capture_manager.stop_capture()
    
    if not raw_frames:
        return None
        
    # Process: Resize -> BGR to RGB -> PIL
    processed = capture_manager.post_process_frames([raw_frames[-1]])
    rgb_frame = cv2.cvtColor(processed[0], cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def main():
    # --- 1. System Initialization ---
    print(">>> [SYSTEM] Initializing Vision...")
    capture_config = SystemConfig(target_fps=30, target_size=(1000, 1000))
    capture_manager = CaptureManager(capture_config)
    capture_manager.start_system()

    print(">>> [SYSTEM] Initializing Controls...")
    controller = InputControllerThread()
    controller.start()

    print(">>> [SYSTEM] Loading Brain (Qwen-VL)...")
    model_init_kwargs = {
        "gpu_memory_utilization": 0.93, 
        "max_model_len": 8000, 
        "trust_remote_code": True
    }
    model_config = BaseModelConfig(
        model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", 
        is_vision_model=True, 
        model_init_kwargs=model_init_kwargs
    )
    llm = JohnVLLM(model_config).load_model(model_config)

    print("\n" + "="*50)
    print(">>> AI AGENT STARTED. PRESS CTRL+C TO STOP.")
    print("="*50 + "\n")

    # --- 2. The Infinite Game Loop ---
    try:
        while True:
            # A. Capture State
            pil_image = capture_latest_frame(capture_manager)
            if pil_image is None:
                print("! Warning: No frame captured. Retrying...")
                time.sleep(0.5)
                continue

            # B. Router: What are we looking at?
            print("\n[ROUTER] Analyzing Game State...")
            state_data = decide_game_state(llm, [pil_image])
            current_state = state_data.get("current_state", "unknown")
            confidence = state_data.get("confidence_score", 0.0)
            reason = state_data.get("reasoning", "No reasoning provided.")
            
            print(f">>> STATE DETECTED: [{current_state.upper()}] (Conf: {confidence:.2f})")
            print(f">>> Reason: {reason}")

            # C. Dispatch to Handler
            if current_state == "main_menu":
                # Handles Start Screen, Settings, Pause Menus
                main_menu_handler(llm, [pil_image], controller)
                
            elif current_state == "dialogue":
                # Handles Text Boxes and Conversation Choices
                dialogue_handler(llm, [pil_image], controller)
                
            elif current_state == "gameplay":
                print("[GAMEPLAY] No specific handler yet. Exploring...")
                # Placeholder: In the future, call gameplay_handler(llm, img, controller)
                # For now, maybe just walk forward a bit?
                # controller.execute_action({"type": "key_press", "details": {"key": "up", "hold_time": 0.5}})
                time.sleep(1) 
                
            elif current_state == "unknown":
                print("[UNKNOWN] Unsure what to do. Waiting...")
                time.sleep(1)

            # D. Loop Rate Limiting
            # Prevent the loop from running too hot if LLM is very fast, 
            # though usually the LLM inference acts as the natural rate limiter.
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n>>> [SYSTEM] Stopping Manual Override...")
    except Exception as e:
        print(f"\n>>> [CRITICAL ERROR] {e}")
        traceback.print_exc()
    finally:
        # --- 3. Cleanup ---
        capture_manager.stop_system()
        controller.stop()
        print(">>> [SYSTEM] Shutdown Complete.")

if __name__ == "__main__":
    main()





# import cv2
# import time
# from PIL import Image

# # Interfaces
# from interfaces.vllm_interface import JohnVLLM
# from interfaces.base import BaseModelConfig
# from gaming.controls import InputControllerThread

# # Local Modules
# from gaming.game_capture import CaptureManager, SystemConfig
# from gaming.main_menu_handler import main_menu_handler
# from gaming.navigation_utils import ensure_menu_selection

# def main():
#     # 1. Setup Systems
#     capture_config = SystemConfig(target_fps=30, target_size=(1000, 1000))
#     capture_manager = CaptureManager(capture_config)
#     capture_manager.start_system()

#     controller = InputControllerThread()
#     controller.start()

#     model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True}
#     model_config = BaseModelConfig(
#         model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", 
#         is_vision_model=True, 
#         model_init_kwargs=model_init_kwargs
#     )
#     llm = JohnVLLM(model_config).load_model(model_config)

#     try:
#         print(">>> System Ready.")
        
#         # --- CRITICAL STEP: Wake up the menu ---
#         # Before we look, we wiggle the controls to ensure a cursor is visible.
#         ensure_menu_selection(controller)
        
#         # Give game a split second to render the cursor highlight
#         time.sleep(0.2) 

#         # 2. Capture Frame
#         capture_manager.start_capture()
#         time.sleep(0.1) # Short grab
#         raw_frames = capture_manager.stop_capture()
        
#         if raw_frames:
#             processed = capture_manager.post_process_frames([raw_frames[-1]])
#             rgb_frame = cv2.cvtColor(processed[0], cv2.COLOR_BGR2RGB)
#             pil_image = Image.fromarray(rgb_frame)

#             # 3. Run Handler
#             main_menu_handler(llm, [pil_image], controller)

#     finally:
#         capture_manager.stop_system()
#         controller.stop()

# if __name__ == "__main__":
#     main()



# ## bbox acc checking
# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread("debug_frames/lacie-test.png")  # Replace with your image path

# # Define the bounding boxes from the JSON
# boxes = [
#     {"label": "player", "bbox": [476, 376, 523, 496], "color": (0, 255, 0)},
#     {"label": "bird", "bbox": [343, 616, 376, 665], "color": (0, 0, 255)},
#     {"label": "birdhouse", "bbox": [208, 442, 243, 616], "color": (255, 0, 0)},
#     # {"label": "tree", "bbox": [100, 273, 267, 513], "color": (0, 255, 255)},
#     {"label": "tree", "bbox": [0, 0, 267, 640], "color": (255, 255, 0)},
#     {"label": "stone_path", "bbox": [434, 256, 565, 636], "color": (128, 0, 255)},
#     {"label": "house", "bbox": [376, 0, 612, 276], "color": (255, 128, 0)},
#     # {"label": "brick_wall", "bbox": [690, 214, 885, 536], "color": (128, 128, 255)},
#     {"label": "fence", "bbox": [706, 650, 862, 773], "color": (0, 128, 255)},
#     {"label": "gate", "bbox": [455, 640, 536, 773], "color": (255, 0, 255)},
#     {"label": "lamp_post", "bbox": [112, 583, 137, 902], "color": (128, 255, 0)},
#     # {"label": "lamp_post", "bbox": [414, 486, 434, 536], "color": (128, 255, 0)},
#     {"label": "lamp_post", "bbox": [562, 552, 583, 636], "color": (128, 255, 0)},
#     # {"label": "lamp_post", "bbox": [894, 507, 918, 686], "color": (128, 255, 0)}
# ]

# # Draw each bounding box
# for box in boxes:
#     x, y, w, h = box["bbox"]
#     color = box["color"]
#     label = box["label"]
#     cv2.rectangle(image, (x, y), (w, h), color, 2)
#     cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# # Save or display the image
# cv2.imwrite("game_screenshot_with_boxes.png", image)
# cv2.imshow("Image with Bounding Boxes", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#############################################################################
# import time
# import os
# import logging
# import cv2
# import numpy as np
# from typing import Optional

# # --- Imports ---
# # 1. The Optimized Capture System (saved from previous step)
# from gaming.game_capture import CaptureManager, SystemConfig

# # 2. Your Input Controller
# try:
#     from gaming.controls import InputControllerThread
# except ImportError:
#     # Mocking for demonstration if the file isn't present locally
#     import threading
#     class InputControllerThread(threading.Thread):
#         def execute_action(self, action): print(f"[MockInput] Executing: {action}")
#         def stop(self): pass
#         def run(self): pass

# # --- Setup Logging ---
# logging.basicConfig(
#     level=logging.INFO, 
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S"
# )


# # hacky temporary fix for running with sudo -- gives permission of captured images to user.
# def save_as_user(path, image):
#     """Saves an image and immediately restores ownership to the non-root user."""
#     # 1. Save the file (currently owned by root)
#     cv2.imwrite(path, image)
    
#     # 2. Check if we are running via sudo
#     sudo_uid = os.environ.get('SUDO_UID')
#     sudo_gid = os.environ.get('SUDO_GID')
    
#     if sudo_uid and sudo_gid:
#         try:
#             # 3. Change ownership back to the original user
#             os.chown(path, int(sudo_uid), int(sudo_gid))
#         except Exception as e:
#             logging.warning(f"Could not change file ownership: {e}")


# def capture_action_sequence():
#     """
#     Orchestrates the Before -> Action -> After capture flow.
#     """
#     # --- Configuration ---
#     ACTION_DURATION = 1.0
#     OUTPUT_DIR = 'debug_frames/character_dataset'
    
#     # Configure the system for VLM (1000x1000)
#     config = SystemConfig(
#         device_index=0,
#         src_width=2560,
#         src_height=1440,
#         target_fps=60,
#         target_size=(1000, 1000), # VLM Standard
#         enable_psutil=True,
#         warmup_time=2.0
#     )

#     manager = CaptureManager(config)
#     input_controller = InputControllerThread()
    
#     try:
#         # 1. Start Background Processes (Includes Warmup)
#         logging.info("System: Initializing workers...")
#         input_controller.start()
#         manager.start_system() # This blocks for 2.0s for warmup
        
#         time.sleep(2.0)
#         # 2. Capture "Before" State
#         logging.info("Phase: Capturing 'Before' frame...")
#         # We capture a tiny slice of time to ensure we get a fresh frame
#         before_frame_raw = manager.get_snapshot()
#         if before_frame_raw is None:
#             raise RuntimeError("Failed to capture 'Before' frame.")

#         # 3. Capture "Action" State
#         logging.info(f"Phase: Executing Action for {ACTION_DURATION}s...")
        
#         # Start filling the RAM buffer
#         manager.start_capture()
        
#         # Trigger the physical action
#         start_t = time.perf_counter()
#         input_controller.execute_action({
#             "type": "key_press",
#             "details": {"key": ["left"], "hold_time": ACTION_DURATION}
#         })
        
#         # Wait strictly for the duration
#         # We calculate sleep to ensure exact timing, accounting for execution overhead
#         elapsed = time.perf_counter() - start_t
#         remaining = ACTION_DURATION - elapsed
#         if remaining > 0:
#             time.sleep(remaining)
            
#         # Stop filling buffer
#         during_frames_raw = manager.stop_capture()
#         logging.info(f"Action captured: {len(during_frames_raw)} raw frames.")

#         # 4. Capture "After" State
#         # Wait a moment for physics/animations to settle
#         time.sleep(0.2)
        
#         logging.info("Phase: Capturing 'After' frame...")
#         after_frame_raw = manager.get_snapshot()
#         if after_frame_raw is None:
#             raise RuntimeError("Failed to capture 'After' frame.")

#         # 5. Post-Processing (The Heavy Lifting)
#         logging.info("Phase: Post-Processing (Resizing & Letterboxing)...")
        
#         # Combine everything into one list to maximize thread pool efficiency
#         # Structure: [Before] + [During...] + [After]
#         all_raw_frames = [before_frame_raw] + during_frames_raw + [after_frame_raw]
        
#         t0 = time.perf_counter()
#         # This runs the 1000x1000 letterbox logic on all cores
#         all_processed = manager.post_process_frames(all_raw_frames)
#         logging.info(f"Processed {len(all_processed)} frames in {time.perf_counter() - t0:.3f}s")

#         # Separate them back out
#         before_final = all_processed[0]
#         during_final = all_processed[1:-1]
#         after_final = all_processed[-1]

#         # 6. Save to Disk
#         # Note: cv2.imwrite expects BGR, which is what we have. No conversion needed.
#         logging.info(f"Saving to {OUTPUT_DIR}...")
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         sudo_uid = os.environ.get('SUDO_UID')
#         sudo_gid = os.environ.get('SUDO_GID')
#         if sudo_uid and sudo_gid:
#             os.chown(OUTPUT_DIR, int(sudo_uid), int(sudo_gid))

#         # Use the helper function instead of cv2.imwrite directly -- TODO: drop when moving on.
#         save_as_user(os.path.join(OUTPUT_DIR, 'capture_before_action.png'), before_final)
#         save_as_user(os.path.join(OUTPUT_DIR, 'capture_after_action.png'), after_final)
        
#         for i, frame in enumerate(during_final):
#             fname = f"during_action_{i:04d}.png"
#             save_as_user(os.path.join(OUTPUT_DIR, fname), frame)
#         logging.info("Sequence complete.")

#     except Exception as e:
#         logging.error(f"Critical Error: {e}", exc_info=True)
#     finally:
#         # Clean shutdown
#         if input_controller.is_alive():
#             input_controller.stop()
#         manager.stop_system()
#         logging.info("System shutdown.")

# if __name__ == "__main__":
#     capture_action_sequence()



# import time
# time.sleep(5)
# take_action()
# ##########################################################################






















#async vllm generator test
# from interfaces.vllm_interface import JohnVLLMAsync
# from interfaces.base import BaseModelConfig


# generation_config = {
#     "max_tokens": 512,
#     "temperature": 0.7,
#     "top_p": 1.0,
#     "top_k": -1,
#     "repetition_penalty": 1.0,
#     "output_kind": "DELTA",
#     # Guided decoding or other complex params can be added here
# }

# import asyncio

# model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
#     }
# model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)

# async def generate():
#     vllm = await JohnVLLMAsync(model_config).load_model(model_config)
#     # async with vllm:

#     async for response in vllm.dialogue_generator(prompt="Hello", generation_config=generation_config):
#         print(response)


# asyncio.run(generate())
