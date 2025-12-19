# import cv2
# import numpy as np
# import os
# import heapq
# import math
# import time
# from collections import deque

# # --- 1. Controller Integration ---
# from gaming.controls import InputControllerThread

# # --- 2. Configuration ---
# VIDEO_SOURCE = 0 
# GRID_SCALE = 20         
# SAFE_DISTANCE = 50      
# HOLD_TIME = 0.1         # Duration of key press
# START_DELAY = 3.0       
# ARRIVAL_THRESHOLD = 40  
# LOOKAHEAD_DIST = 40     

# # STUCK DETECTION SETTINGS
# MIN_MOVEMENT_THRESHOLD = 3.0  # Pixels. If moved less than this, we are "stuck"
# STUCK_FRAME_LIMIT = 3         # How many consecutive actions can fail before we blacklist
# BLACKLIST_DURATION = 10.0      # How long (seconds) to remember an invisible wall

# MODEL_FILES = {
#     "model": "Sagex/gaming/dasiamrpn_model.onnx",
#     "kernel": "Sagex/gaming/dasiamrpn_kernel_r1.onnx",
#     "cls": "Sagex/gaming/dasiamrpn_kernel_cls1.onnx"
# }

# # --- 3. Advanced Pathfinding with Blacklisting ---
# class PathFinder:
#     def __init__(self, width, height, scale, padding):
#         self.width, self.height = width, height
#         self.scale, self.padding = scale, padding
#         self.grid_w, self.grid_h = width // scale, height // scale

#     def heuristic(self, a, b):
#         return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

#     def get_blocked_cells(self, obstacle_boxes, temp_blacklist):
#         blocked = set()
        
#         # 1. Add Known Visual Obstacles (Red/Orange Boxes)
#         for box in obstacle_boxes:
#             x, y, w, h = [int(v) for v in box]
#             x_s, y_s = x - self.padding, y - self.padding
#             w_s, h_s = w + (self.padding * 2), h + (self.padding * 2)
            
#             gx_s = max(0, x_s // self.scale)
#             gy_s = max(0, y_s // self.scale)
#             gx_e = min(self.grid_w - 1, (x_s + w_s) // self.scale)
#             gy_e = min(self.grid_h - 1, (y_s + h_s) // self.scale)

#             for gx in range(int(gx_s), int(gx_e) + 1):
#                 for gy in range(int(gy_s), int(gy_e) + 1):
#                     blocked.add((gx, gy))

#         # 2. Add Temporary "Invisible" Obstacles (Stuck spots)
#         current_time = time.time()
#         for cell, expire_time in temp_blacklist.items():
#             if current_time < expire_time:
#                 blocked.add(cell)
                
#         return blocked

#     def find_nearest_safe_cell(self, start_node, blocked_cells):
#         """BFS: Finds the closest grid cell NOT in blocked_cells"""
#         queue = deque([start_node])
#         visited = {start_node}
        
#         while queue:
#             curr = queue.popleft()
#             if curr not in blocked_cells:
#                 return curr # Found safety
            
#             # Search neighbors
#             for dx, dy in [(0,1),(1,0),(0,-1),(-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
#                 nxt = (curr[0]+dx, curr[1]+dy)
#                 if 0 <= nxt[0] < self.grid_w and 0 <= nxt[1] < self.grid_h:
#                     if nxt not in visited:
#                         visited.add(nxt)
#                         queue.append(nxt)
#         return None

#     def solve(self, start_pix, end_pix, obstacle_boxes, temp_blacklist):
#         start = (start_pix[0] // self.scale, start_pix[1] // self.scale)
#         goal = (end_pix[0] // self.scale, end_pix[1] // self.scale)
        
#         blocked_cells = self.get_blocked_cells(obstacle_boxes, temp_blacklist)

#         # --- ESCAPE MODE ---
#         if start in blocked_cells:
#             safe_node = self.find_nearest_safe_cell(start, blocked_cells)
#             if safe_node:
#                 safe_pix = (safe_node[0]*self.scale + self.scale//2, safe_node[1]*self.scale + self.scale//2)
#                 # Return the specific target NODE too, so we can blacklist it if we get stuck
#                 return ["ESCAPE", safe_pix, safe_node]
#             else:
#                 return None # Totally trapped

#         # --- PATH MODE (A*) ---
#         frontier = []
#         heapq.heappush(frontier, (0, start))
#         came_from, cost_so_far = {start: None}, {start: 0}

#         while frontier:
#             _, current = heapq.heappop(frontier)
#             if current == goal: break
            
#             for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
#                 nxt = (current[0]+dx, current[1]+dy)
#                 if 0 <= nxt[0] < self.grid_w and 0 <= nxt[1] < self.grid_h:
#                     if nxt not in blocked_cells:
#                         cost = 1.414 if dx!=0 and dy!=0 else 1.0
#                         new_cost = cost_so_far[current] + cost
#                         if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
#                             cost_so_far[nxt] = new_cost
#                             priority = new_cost + self.heuristic(goal, nxt)
#                             heapq.heappush(frontier, (priority, nxt))
#                             came_from[nxt] = current

#         if goal not in came_from: return None

#         path = []
#         curr = goal
#         while curr != start:
#             path.append((curr[0] * self.scale + self.scale//2, curr[1] * self.scale + self.scale//2))
#             curr = came_from[curr]
#         path.reverse()
#         return ["PATH", path]

# # --- 4. Main Application ---

# def create_tracker():
#     params = cv2.TrackerDaSiamRPN_Params()
#     params.model = MODEL_FILES["model"]
#     params.kernel_r1 = MODEL_FILES["kernel"]
#     params.kernel_cls1 = MODEL_FILES["cls"]
#     return cv2.TrackerDaSiamRPN_create(params)

# def get_center(bbox):
#     return (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

# def determine_keys(curr, target, threshold=10):
#     keys = []
#     dx, dy = target[0] - curr[0], target[1] - curr[1]
#     if dy < -threshold: keys.append('up')
#     elif dy > threshold: keys.append('down')
#     if dx < -threshold: keys.append('left')
#     elif dx > threshold: keys.append('right')
#     return keys

# def get_lookahead_point(path, current_pos, lookahead_dist):
#     if not path: return current_pos
#     for point in path:
#         if math.dist(current_pos, point) > lookahead_dist:
#             return point
#     return path[-1]

# def main():
#     if not all(os.path.exists(v) for v in MODEL_FILES.values()):
#         print("[ERROR] Check model paths."); return

#     controller = InputControllerThread()
#     controller.start()
#     cap = cv2.VideoCapture(VIDEO_SOURCE)
#     ret, frame = cap.read()
#     if not ret: return

#     # Setup
#     print("Setup: CHAR -> OBJ -> OBS (ESC)")
#     c_b = cv2.selectROI("Setup", frame, False)
#     o_b = cv2.selectROI("Setup", frame, False)
#     obs = []
#     while True:
#         b = cv2.selectROI("Setup", frame, False)
#         if b[2] == 0: break
#         obs.append(b)
#     cv2.destroyWindow("Setup")

#     # Trackers
#     tr_char = create_tracker(); tr_char.init(frame, c_b)
#     tr_obj = create_tracker(); tr_obj.init(frame, o_b)
#     tr_obs = [create_tracker() for _ in range(len(obs))]
#     for i, t in enumerate(tr_obs): t.init(frame, obs[i])

#     pf = PathFinder(frame.shape[1], frame.shape[0], GRID_SCALE, SAFE_DISTANCE)
#     start_time = time.time()
    
#     # Logic State Variables
#     reached = False
#     next_action_time = 0 
    
#     # Stuck Detection Variables
#     last_position = get_center(c_b)
#     stuck_frames_count = 0
#     temp_blacklist = {} # Format: {(grid_x, grid_y): expiration_timestamp}
#     current_target_node = None # The grid node we are currently trying to reach

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         # 1. Update Tracking
#         _, c_b = tr_char.update(frame)
#         _, o_b = tr_obj.update(frame)
#         obs_b = [t.update(frame)[1] for t in tr_obs]
        
#         c_cen = get_center(c_b)
#         o_cen = get_center(o_b)

#         # 2. Visualization
#         for b in obs_b:
#             x,y,w,h = [int(v) for v in b]
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
#             cv2.rectangle(frame, (x-SAFE_DISTANCE, y-SAFE_DISTANCE), (x+w+SAFE_DISTANCE, y+h+SAFE_DISTANCE), (0,165,255), 1)

#         # Draw Blacklisted Zones (Invisible Walls)
#         current_time = time.time()
#         for cell, expire in list(temp_blacklist.items()):
#             if current_time > expire:
#                 del temp_blacklist[cell]
#             else:
#                 # Draw grey X on blacklisted cells
#                 bx, by = cell[0]*GRID_SCALE, cell[1]*GRID_SCALE
#                 cv2.rectangle(frame, (bx, by), (bx+GRID_SCALE, by+GRID_SCALE), (100,100,100), -1)

#         elapsed = time.time() - start_time
#         dist_to_goal = math.dist(c_cen, o_cen)

#         if dist_to_goal < ARRIVAL_THRESHOLD:
#             reached = True
#             cv2.putText(frame, "REACHED", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
#         elif not reached and elapsed > START_DELAY:
#             if current_time > next_action_time:
                
#                 # --- A. STUCK DETECTION ---
#                 # Compare current position to position from LAST command
#                 move_dist = math.dist(c_cen, last_position)
                
#                 # If we moved less than threshold, we might be stuck
#                 if move_dist < MIN_MOVEMENT_THRESHOLD:
#                     stuck_frames_count += 1
#                     # print(f"Stuck Warning: {stuck_frames_count}/{STUCK_FRAME_LIMIT}")
#                 else:
#                     stuck_frames_count = 0 # Reset if we moved successfully

#                 # Check if we hit the limit
#                 if stuck_frames_count >= STUCK_FRAME_LIMIT:
#                     print("[WARN] Character Stuck! Blacklisting target node.")
                    
#                     # Blacklist the node we were trying to reach
#                     if current_target_node:
#                         temp_blacklist[current_target_node] = current_time + BLACKLIST_DURATION
#                         cv2.putText(frame, "STUCK! REROUTING...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
#                     stuck_frames_count = 0 # Reset counter after taking action

#                 # Update history
#                 last_position = c_cen

#                 # --- B. PATHFINDING ---
#                 result = pf.solve(c_cen, o_cen, obs_b, temp_blacklist)

#                 if result:
#                     mode = result[0] # "ESCAPE" or "PATH"
                    
#                     if mode == "ESCAPE":
#                         target_point = result[1]
#                         current_target_node = result[2] # Store grid node for blacklisting
                        
#                         cv2.putText(frame, "!!! ESCAPING !!!", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#                         cv2.line(frame, c_cen, target_point, (0,0,255), 3)
                        
#                         keys = determine_keys(c_cen, target_point)

#                     else: # mode == "PATH"
#                         path = result[1]
#                         # Draw Path
#                         for i in range(len(path)-1): 
#                             cv2.line(frame, path[i], path[i+1], (255,255,0), 2)

#                         # Lookahead
#                         target_point = get_lookahead_point(path, c_cen, LOOKAHEAD_DIST)
                        
#                         # Calculate grid node of target point (approx) for blacklisting purposes
#                         current_target_node = (target_point[0]//GRID_SCALE, target_point[1]//GRID_SCALE)
                        
#                         cv2.arrowedLine(frame, c_cen, target_point, (0,255,255), 3)
#                         keys = determine_keys(c_cen, target_point)

#                     # --- C. EXECUTION ---
#                     if keys:
#                         controller.execute_action({
#                             "type": "key_press",
#                             "details": {"key": keys, "hold_time": HOLD_TIME}
#                         })
#                         next_action_time = current_time + HOLD_TIME
#                         cv2.putText(frame, f"CMD: {keys}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
#                 else:
#                     cv2.putText(frame, "NO PATH POSSIBLE", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#             else:
#                 # Waiting for cooldown
#                 pass

#         elif elapsed <= START_DELAY:
#             cv2.putText(frame, f"START: {int(START_DELAY-elapsed)}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

#         cv2.rectangle(frame, (int(c_b[0]), int(c_b[1])), (int(c_b[0]+c_b[2]), int(c_b[1]+c_b[3])), (0,255,0), 2)
#         cv2.imshow("Robust Navigation", frame)
#         if cv2.waitKey(1) == ord('q'): break

#     controller.stop()
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



import cv2
import math
import time
import os
import heapq
from collections import deque
from typing import List, Tuple, Optional, Any

# Internal Imports
from gaming.gameplay_structs import GameplayScan, GameEntity

# --- Configuration ---
GRID_SCALE = 20         
SAFE_DISTANCE = 40      
HOLD_TIME = 0.08        # Fast taps for responsiveness
ARRIVAL_THRESHOLD = 35  
LOOKAHEAD_DIST = 50     

# Stuck Detection
MIN_MOVEMENT_THRESHOLD = 2.0 
STUCK_FRAME_LIMIT = 5         
BLACKLIST_DURATION = 8.0      

MODEL_FILES = {
    "model": "Sagex/gaming/dasiamrpn_model.onnx",
    "kernel": "Sagex/gaming/dasiamrpn_kernel_r1.onnx",
    "cls": "Sagex/gaming/dasiamrpn_kernel_cls1.onnx"
}

class PathFinder:
    def __init__(self, width, height, scale, padding):
        self.width, self.height = width, height
        self.scale, self.padding = scale, padding
        self.grid_w, self.grid_h = width // scale, height // scale

    def heuristic(self, a, b):
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_blocked_cells(self, obstacle_boxes, temp_blacklist):
        blocked = set()
        # 1. Visual Obstacles
        for box in obstacle_boxes:
            x, y, w, h = [int(v) for v in box]
            # Add padding
            x_s, y_s = max(0, x - self.padding), max(0, y - self.padding)
            x_e, y_e = x + w + self.padding, y + h + self.padding
            
            gx_s, gy_s = x_s // self.scale, y_s // self.scale
            gx_e, gy_e = x_e // self.scale, y_e // self.scale
            
            for gx in range(int(gx_s), int(gx_e) + 1):
                for gy in range(int(gy_s), int(gy_e) + 1):
                    blocked.add((gx, gy))

        # 2. Invisible/Stuck Walls
        current_time = time.time()
        for cell, expire_time in temp_blacklist.items():
            if current_time < expire_time:
                blocked.add(cell)
        return blocked

    def solve(self, start_pix, end_pix, obstacle_boxes, temp_blacklist):
        start = (int(start_pix[0] // self.scale), int(start_pix[1] // self.scale))
        goal = (int(end_pix[0] // self.scale), int(end_pix[1] // self.scale))
        
        blocked = self.get_blocked_cells(obstacle_boxes, temp_blacklist)

        # A* Search
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from, cost_so_far = {start: None}, {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal: break
            
            # 8-Directional movement
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nxt = (current[0]+dx, current[1]+dy)
                
                # Bounds check
                if 0 <= nxt[0] < self.grid_w and 0 <= nxt[1] < self.grid_h:
                    if nxt not in blocked:
                        weight = 1.414 if dx!=0 and dy!=0 else 1.0
                        new_cost = cost_so_far[current] + weight
                        if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                            cost_so_far[nxt] = new_cost
                            priority = new_cost + self.heuristic(goal, nxt)
                            heapq.heappush(frontier, (priority, nxt))
                            came_from[nxt] = current
        
        # Reconstruct
        if goal not in came_from:
            # Fallback: Find nearest unblocked node to current position (Escape)
            return None 

        path = []
        curr = goal
        while curr != start:
            path.append((curr[0] * self.scale + self.scale//2, curr[1] * self.scale + self.scale//2))
            curr = came_from[curr]
        path.reverse()
        return path

class MovementAgent:
    def __init__(self, controller):
        self.controller = controller
        self._check_models()

    def _check_models(self):
        if not all(os.path.exists(v) for v in MODEL_FILES.values()):
            print("[ERROR] Tracker models missing. Movement will fail.")

    def _create_tracker(self):
        params = cv2.TrackerDaSiamRPN_Params()
        params.model = MODEL_FILES["model"]
        params.kernel_r1 = MODEL_FILES["kernel"]
        params.kernel_cls1 = MODEL_FILES["cls"]
        return cv2.TrackerDaSiamRPN_create(params)

    def _get_center(self, bbox):
        return (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

    def _determine_keys(self, curr, target):
        keys = []
        dx, dy = target[0] - curr[0], target[1] - curr[1]
        if dy < -10: keys.append('up')
        elif dy > 10: keys.append('down')
        if dx < -10: keys.append('left')
        elif dx > 10: keys.append('right')
        return keys

    def execute_navigation(self, capture_manager, scan_data: GameplayScan):
        """
        Main Loop: Tracks objects defined in scan_data and moves character to target.
        """
        print(f"Movement: Initializing. Goal -> {scan_data.target.label}")

        # 1. Grab Initial Frame
        capture_manager.start_capture()
        time.sleep(0.1)
        raw_frames = capture_manager.stop_capture()
        if not raw_frames: return "capture_error"
        
        # Process to standard size (must match VLM coordinate space 1000x1000)
        frame = capture_manager.post_process_frames([raw_frames[-1]])[0]
        
        # 2. Initialize Trackers
        try:
            tr_char = self._create_tracker()
            tr_char.init(frame, scan_data.get_cv2_rect(scan_data.character))
            
            tr_target = self._create_tracker()
            tr_target.init(frame, scan_data.get_cv2_rect(scan_data.target))
            
            tr_obs = []
            for obs in scan_data.obstacles:
                t = self._create_tracker()
                t.init(frame, scan_data.get_cv2_rect(obs))
                tr_obs.append(t)
        except Exception as e:
            print(f"Movement: Tracker Init Failed. {e}")
            return "tracker_error"

        # 3. Setup Logic
        pf = PathFinder(frame.shape[1], frame.shape[0], GRID_SCALE, SAFE_DISTANCE)
        last_pos = self._get_center(scan_data.get_cv2_rect(scan_data.character))
        stuck_count = 0
        temp_blacklist = {}
        
        start_time = time.time()
        last_action_time = 0

        # --- NAVIGATION LOOP ---
        try:
            while True:
                # Refresh Frame
                capture_manager.start_capture()
                time.sleep(0.02) # Fast cycle
                raw = capture_manager.stop_capture()
                if not raw: continue
                frame = capture_manager.post_process_frames([raw[-1]])[0]

                # Update Trackers
                ok_c, box_c = tr_char.update(frame)
                ok_t, box_t = tr_target.update(frame)
                boxes_obs = [t.update(frame)[1] for t in tr_obs]
                
                if not ok_c or not ok_t:
                    print("Movement: Lost tracking.")
                    return "lost_tracking"

                c_cen = self._get_center(box_c)
                t_cen = self._get_center(box_t)

                # Visualization (Optional, for debug view)
                cv2.rectangle(frame, (int(box_c[0]), int(box_c[1])), (int(box_c[0]+box_c[2]), int(box_c[1]+box_c[3])), (0,255,0), 2)
                cv2.circle(frame, t_cen, 5, (0,0,255), -1)

                # Check Arrival
                dist = math.dist(c_cen, t_cen)
                if dist < ARRIVAL_THRESHOLD:
                    print("Movement: Target Reached!")
                    return "success"

                # Navigation Logic (Rate Limited)
                current_time = time.time()
                if current_time - last_action_time > HOLD_TIME:
                    
                    # Stuck Detection
                    if math.dist(c_cen, last_pos) < MIN_MOVEMENT_THRESHOLD:
                        stuck_count += 1
                    else:
                        stuck_count = 0
                    last_pos = c_cen

                    if stuck_count > STUCK_FRAME_LIMIT:
                        print("Movement: STUCK detected. Rerouting...")
                        # Block the immediate area in front of player
                        grid_node = (c_cen[0]//GRID_SCALE, c_cen[1]//GRID_SCALE)
                        temp_blacklist[grid_node] = current_time + BLACKLIST_DURATION
                        stuck_count = 0

                    # Pathfind
                    path = pf.solve(c_cen, t_cen, boxes_obs, temp_blacklist)
                    
                    if path:
                        # Lookahead
                        target_point = path[0]
                        for pt in path:
                            if math.dist(c_cen, pt) > LOOKAHEAD_DIST:
                                target_point = pt
                                break
                        
                        # Execute
                        keys = self._determine_keys(c_cen, target_point)
                        if keys:
                            self.controller.execute_action({
                                "type": "key_press", 
                                "details": {"key": keys, "hold_time": HOLD_TIME}
                            })
                            last_action_time = current_time
                    else:
                        print("Movement: No path found.")
                        # Simple wiggle to unstuck?
                        
                # Timeout safety
                if current_time - start_time > 15.0:
                    print("Movement: Timed out.")
                    return "timeout"
                    
        except KeyboardInterrupt:
            return "interrupted"