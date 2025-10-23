import pygame
import sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer, AerSimulator
# from qiskit_ibm_runtime.fake_provider import FakePerth
import numpy as np
from qiskit.visualization import plot_histogram
import time
from enum import Enum
import threading
import random
import math
from qiskit_algorithms import AmplificationProblem
from qiskit_algorithms import Grover
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZGate


# --- Backend Code (Modified for GUI) ---
#backend = AerSimulator.from_backend(FakePerth())

class QustomerStatus(Enum):
    IN_LINE = 0
    WAITING = 1
    COMPLETE = 2

class GameController:
    debug_on = False # Set to False to enable timers
    order_queue = []
    pickup_queue = []
    ready_food = {}
    cur_qustomer_id = 1
    points = 0
    strikes = 0

    log = ["Welcome to the Quantum Cafe!"] # GUI will display this
    lock = threading.RLock() # For thread-safe access to lists/dicts
    game_over = False

    @classmethod
    def log_message(cls, message):
        """Adds a message to the game log for the GUI."""
        with cls.lock:
            cls.log.append(message)
            if len(cls.log) > 15: # Keep log from getting too long
                cls.log.pop(0)
        print(message) # Also print to console for debugging

    # These show_ methods are no longer needed, GUI will render directly
    @classmethod
    def show_order_queue(cls):
        pass

    @classmethod
    def show_pickup_queue(cls):
        pass
    
    @classmethod
    def show_ready_food(cls):
        pass
    
    @classmethod
    def show_game_state(cls):
        pass

    @classmethod
    def set_timer(cls, qustomer, seconds):
        # --- MODIFIED FOR TIMER FIX ---
        qustomer.timer_start_time = time.time()
        qustomer.timer_duration = seconds
        qustomer.timer_active = True
        
        with cls.lock:
            qustomer.timer_id += 1 # Invalidate any old timer threads
            current_timer_id = qustomer.timer_id
        # --- END MODIFIED ---
        
        if not cls.debug_on:
            # Pass the current_timer_id to the new thread
            timer_thread = threading.Thread(target=qustomer.start_timer, args=(seconds, qustomer.status, current_timer_id))
            timer_thread.daemon = True
            timer_thread.start()

    @classmethod
    def qustomer_enter(cls, qustomer,  counte):
        cls.log_message("Enter qustomer #" + str(qustomer.id))
        with cls.lock:
            cls.order_queue.append(qustomer)
        cls.set_timer(qustomer, 15) # Increased time for GUI
        for i in range(1, qustomer.n + 1):
            qustomer.qc.h(i)
        if qustomer.entangled:
            if not qustomer.maximal:
                for i in range(1, qustomer.n + 1):
                    qustomer.qc.x(qustomer.n + i)
            for i in range(1, qustomer.n + 1):
                qustomer.qc.cx(i, qustomer.n + i)
            cls.log_message("Created " + ("maximally" if qustomer.maximal else "minimally") + " entangled qustomer #" + str(qustomer.id) + ".5")
            
    @classmethod
    def take_order(cls):
        with cls.lock:
            if not cls.order_queue:
                cls.log_message("No one is in the order queue.")
                return
            qustomer = cls.order_queue.pop(0)
            cls.pickup_queue.append(qustomer)
        
        cls.log_message("Order " + qustomer.order + " for qustomer #" + str(qustomer.id) + " taken.")
        qustomer.status = QustomerStatus.WAITING
        cls.set_timer(qustomer, 30) # This will now correctly start a new timer
    
    @classmethod
    def valid_dish(cls, menu_item):
        if len(menu_item) != 3 or any(c not in '01' for c in menu_item) or menu_item == '111':
            return False
        return True
    
    @classmethod
    def prepare_dish(cls, menu_item):
        if not cls.valid_dish(menu_item):
            cls.log_message(f"Invalid dish format: {menu_item}. Use 3 bits (e.g., '010').")
            return False
        
        with cls.lock:
            if not menu_item in cls.ready_food:
                cls.ready_food[menu_item] = 0
            cls.ready_food[menu_item] += 1
        
        cls.log_message(f"Prepared dish: {menu_item}")
        return True
    
    @classmethod
    def state_to_unitary(cls, psi):
        n = int(np.log2(len(psi)))
        zero = np.zeros_like(psi)
        zero[0] = 1
        v = psi - zero
        v = v / np.linalg.norm(v)
        H = np.eye(2**n) - 2 * np.outer(v, np.conj(v))
        return H

    @classmethod
    def search_ready_subset(cls, qustomer):
        n = qustomer.n
        all_states = 2**n

        # target start (qustomer order)
        # set this to 111 if qustomer has "surprise me" order (grover's algo with result in uniform superposition)
        target_state = qustomer.order if "surprise me" in qustomer.order else '111'
        M = 1

        # ready food
        subset_bitstrings = list(cls.ready_food.keys())
        k = len(subset_bitstrings)

        print(f"Search Space (k): {k} items {subset_bitstrings}")
        print(f"Target (M): {M} item '{target_state}'\n")

        # oracle that marks target
        oracle = QuantumCircuit(n, name="U_f")
        mcz = ZGate().control(n - 1)
        target_reversed = target_state[::-1]
        for i in range(n):
            if target_reversed[i] == '0':
                oracle.x(i)
        oracle.append(mcz, list(range(n)))
        for i in range(n):
            if target_reversed[i] == '0':
                oracle.x(i)
                
        # state preperation: the pool to examine
        target_vector = np.zeros(all_states)
        for bitstring in subset_bitstrings:
            index = int(bitstring, 2)
            target_vector[index] = 1 / np.sqrt(k)
        Um_init = QuantumCircuit(n)
        Um_init.initialize(target_vector, range(n))
        Um_decomposed = Um_init.decompose()
        Um = QuantumCircuit(n, name="Um (prepares k=7)")
        for instruction in Um_decomposed.data:
            if instruction.operation.name != 'reset':
                Um.append(instruction)

        # calculate number of iterations
        R = int(np.floor((np.pi / 4) * np.sqrt(k / M)))
        print(f"Optimal Iterations (R): {R}\n")

        # execute amplification problem
        problem = AmplificationProblem(
            oracle=oracle,
            state_preparation=Um,
            is_good_state=None # We use the oracle, so this isn't needed
        )

        # run grover
        grover = Grover(
            sampler=StatevectorSampler(),
            iterations=R # Use our calculated 2 iterations
        )
        result = grover.amplify(problem) # idk how this works actually

        # debug
        print(f"Top measurement: {result.top_measurement}")
        formatted_counts = {}
        for bitstring, prob in result.circuit_results[0].items():
            formatted_counts[bitstring] = prob
        # plot_histogram(formatted_counts)
        return result.top_measurement
    
    @classmethod
    def prepare_recipe_state(cls, qustomer_order):
        # 1) From the binary bitstring order, identify the menu item from recipes dict
        recipes_key = {
             "000": ["apple", "flour", "sugar", "cinnamon", "butter"],
             "001": ["pumpkin", "flour", "sugar", "eggs", "cinnamon"],
             "011": ["coffee beans", "water"],
             "100": ["chai", "milk", "sugar", "cinnamon"],
             "101": ["pumpkin", "milk", "coffee beans", "sugar", "cinnamon"],
             "110": ["flour", "sugar", "cinnamon", "butter", "eggs"],
             "010": ["banana", "flour", "sugar", "eggs", "butter"]
         } # apple pie, pumpkin pie, coffee, chai latte, pumpkin spice latte, cinnamon roll, banana bread
        
        # 2) Represent each recipe as the (initial/expected) quantum state
        n = len(qustomer_order)
        qc = QuantumCircuit(n)

    
        # 3) measure the qua
        
        return qc

    @classmethod
    def quantum_state_tomography(cls, qustomer):
        ingredients = { "flavor": ["apple", "pumpkin", "cinnamon", "coffee", "chai", "banana"],
                        "dry ingredients": ["flour", "sugar", "baking powder"],
                        "wet ingredients": ["milk", "eggs", "butter"] }
        
        pass



    @classmethod
    def serve_food(cls, qustomer_index):
        if not qustomer_index.isdigit():
            cls.log_message(f"Invalid qustomer index: {qustomer_index}. Must be a number.")
            return False

        qustomer_to_serve = None
        qustomer_id = int(qustomer_index)
        
        with cls.lock:
            for i, qustomer in enumerate(cls.pickup_queue):
                if qustomer.id == qustomer_id:
                    qustomer_to_serve = cls.pickup_queue.pop(i)
                    break
        
        if not qustomer_to_serve:
            cls.log_message(f"No qustomer with ID #{qustomer_id} in pickup queue.")
            return False

        # This prevents a crash in the reverted search_ready_subset (k=0)
        with cls.lock:
            if not cls.ready_food:
                cls.log_message("No food is ready to serve!")
                cls.pickup_queue.append(qustomer_to_serve) # Put back
                return False

        # Run the search
        selected_food = cls.search_ready_subset(qustomer_to_serve)

        # Consume the food item
        with cls.lock:
            # Check if food still exists (could have been used by another thread)
            if selected_food not in cls.ready_food:
                cls.log_message(f"Food {selected_food} was just used! Try again.")
                cls.pickup_queue.append(qustomer_to_serve) # Put back
                return False
                
            cls.ready_food[selected_food] -= 1
            if cls.ready_food[selected_food] == 0:
                del cls.ready_food[selected_food]

        cls.log_message(f"Served {selected_food} to qustomer #{qustomer_id} (wanted {qustomer_to_serve.order})")

        with cls.lock:
            if str(selected_food) == str(qustomer_to_serve.order) or "surprise me" in qustomer_to_serve.order:
                cls.log_message("Order served correctly!")
                cls.points += 1
                if cls.points >= 10:
                    cls.log_message("Congratulations! You have successfully served 10 correct orders and won the game!")
                    cls.game_over = True
            else:
                cls.log_message("Order served incorrectly </3")
                cls.points -= 1
                cls.issue_strike(qustomer_to_serve)
        
        qustomer_to_serve.status = QustomerStatus.COMPLETE
        qustomer_to_serve.timer_active = False # --- ADDED: Stop timer bar on serve ---
        return True
    
    @classmethod
    def issue_strike(cls, qustomer):
        with cls.lock:
            cls.strikes += 1
            cls.log_message(f"Issued strike to qustomer #{qustomer.id}. Total strikes: {cls.strikes}")
            if cls.strikes >= 3:
                cls.log_message("Game over! You have made 3 incorrect orders.")
                cls.game_over = True
    
    @classmethod
    def handle_inline_fail(cls, qustomer):
        try:
            with cls.lock:
                cls.order_queue.remove(qustomer)
                cls.issue_strike(qustomer)
            cls.log_message(f"Qustomer #{qustomer.id} left before ordering.")
        except ValueError:
            pass # Qustomer was already served or removed

    @classmethod
    def handle_waiting_fail(cls, qustomer):
        try:
            with cls.lock:
                cls.pickup_queue.remove(qustomer)
            cls.log_message(f"Qustomer #{qustomer.id} left before food was served.")
        except ValueError:
            pass

    @classmethod
    def spawn_qustomer(cls, n, min_interval, max_interval):
        while not cls.game_over:
            time.sleep(np.random.randint(min_interval, max_interval))
            if cls.game_over: break # Stop spawning if game ended
            
            # Create a new qustomer
            entangled = random.choice([True, False])
            surprise = random.choice([True, False])
            # Run constructor in a thread to avoid blocking spawner
            threading.Thread(target=Qustomer, args=(n, entangled, surprise), daemon=True).start()
            

class Qustomer: 
    def __init__(self, n, entangled=False, surprise_me=False):
        self.n = n
        self.order = ""
        self.entangled = entangled
        if not surprise_me:
            while True:
                self.order = ""
                for i in range(n):
                    self.order += str(np.random.randint(0, 2))
                if self.order != "111":
                    break
        else:
            self.order = "surprise me <( • _ • )>"
        
        with GameController.lock:
            self.id = GameController.cur_qustomer_id
            GameController.cur_qustomer_id += 1
        
        # --- MODIFIED FOR TIMER FIX ---
        self.timer_start_time = 0.0
        self.timer_duration = 0.0
        self.timer_active = False
        self.timer_id = 0 # Unique ID for each timer thread
        # --- END MODIFIED ---

        self.status = QustomerStatus.IN_LINE
        if entangled:
            self.qc = QuantumCircuit(2 * n + 1, n)
            self.maximal = random.randint(0, 1) > 0.5 # maximally or minimally entangles a pair of qustomers
        else:
            self.qc = QuantumCircuit(n + 1, n)
        
        GameController.log_message(f"Qustomer #{self.id} created (Order: {self.order})")
        GameController.qustomer_enter(self, entangled)
    
    # --- MODIFIED FOR TIMER FIX ---
    def start_timer(self, seconds, status_when_started, timer_id):
        start_time = time.time()
        while time.time() - start_time < seconds:
            with GameController.lock:
                # If timer_id changed, a new timer was set. This thread is obsolete.
                if self.timer_id != timer_id:
                    return 
                # If customer is complete, timer is void
                if self.status == QustomerStatus.COMPLETE:
                    self.timer_active = False
                    return
            time.sleep(0.1)
        
        # Time's up. Check if we are still the active timer and not complete.
        with GameController.lock:
            if self.timer_id != timer_id or self.status == QustomerStatus.COMPLETE:
                return
        
        self.timer_active = False # Stop the bar
        
        # Only trigger fail state if status is still what it was when timer started
        if self.status == status_when_started:
            if self.status == QustomerStatus.IN_LINE:
                GameController.handle_inline_fail(qustomer = self)
            elif self.status == QustomerStatus.WAITING:
                GameController.handle_waiting_fail(qustomer = self)
    # --- END MODIFIED ---

# --- Pygame GUI Code ---
# (No changes to GUI code, so it is omitted for brevity)
# (Please use the GUI code from the previous response)

# Helper class for simple buttons
class Button:
    def __init__(self, rect, text, bg_color, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.bg_color = bg_color
        self.text_color = text_color

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=5)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Helper class for simple text input boxes
class InputBox:
    def __init__(self, rect, text_color, bg_color=(255, 255, 255), active_color=(200, 200, 255)):
        self.rect = pygame.Rect(rect)
        self.text_color = text_color
        self.bg_color = bg_color
        self.active_color = active_color
        self.text = ""
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                return "enter" # Signal to process
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode
        return None

    def draw(self, screen, font):
        color = self.active_color if self.active else self.bg_color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, self.text_color, self.rect, 2) # Border
        text_surf = font.render(self.text, True, self.text_color)
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))

# Helper function to draw text
def draw_text(screen, text, font, pos, color=(0, 0, 0)):
    text_surf = font.render(text, True, color)
    screen.blit(text_surf, pos)

def run_game():
    pygame.init()
    
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Cafe")
    clock = pygame.time.Clock()

    # Fonts
    title_font = pygame.font.Font(None, 40)
    main_font = pygame.font.Font(None, 32)
    log_font = pygame.font.Font(None, 24)

    # Colors
    COLOR_BG = (245, 245, 220) # Beige
    COLOR_TITLE = (80, 40, 0) # Brown
    COLOR_TEXT = (0, 0, 0)
    COLOR_STRIKE = (220, 20, 60) # Crimson
    COLOR_SUCCESS = (0, 128, 0) # Green
    
    COLOR_BAR_BG = (50, 50, 50)
    COLOR_BAR_HI = (0, 200, 0)
    COLOR_BAR_MED = (255, 255, 0)
    COLOR_BAR_LOW = (200, 0, 0)
    BAR_MAX_WIDTH = 300
    BAR_HEIGHT = 15

    # --- Initialize GUI Elements ---
    btn_take_order = Button((550, 400, 200, 50), "Take Order", (0, 150, 0))
    
    input_qook = InputBox((550, 470, 200, 50), COLOR_TEXT)
    btn_qook = Button((780, 470, 200, 50), "Let's Qook", (150, 0, 0))
    
    input_serve = InputBox((550, 540, 200, 50), COLOR_TEXT)
    btn_serve = Button((780, 540, 200, 50), "Serve", (0, 0, 150))

    # --- Start Backend Logic ---
    n = 3 # 2^n menu items
    # Create initial qustomers in threads so they don't block GUI
    threading.Thread(target=Qustomer, args=(n, True, True), daemon=True).start()
    time.sleep(0.1) # Stagger
    threading.Thread(target=Qustomer, args=(n,), daemon=True).start()
    
    if not GameController.debug_on:
        spawn_thread = threading.Thread(target=GameController.spawn_qustomer, args=(n, 5, 12)) # Slower spawn for GUI
        spawn_thread.daemon = True
        spawn_thread.start()

    # --- Main Game Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not GameController.game_over:
                # Handle input box events
                input_qook.handle_event(event)
                input_serve.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_take_order.is_clicked(event.pos):
                        threading.Thread(target=GameController.take_order, daemon=True).start()
                    
                    elif btn_qook.is_clicked(event.pos):
                        item = input_qook.text
                        if item:
                            threading.Thread(target=GameController.prepare_dish, args=(item,), daemon=True).start()
                            input_qook.text = ""
                            input_qook.active = False

                    elif btn_serve.is_clicked(event.pos):
                        idx = input_serve.text
                        if idx:
                            threading.Thread(target=GameController.serve_food, args=(idx,), daemon=True).start()
                            input_serve.text = ""
                            input_serve.active = False

        # --- Drawing ---
        screen.fill(COLOR_BG)
        
        # --- Get Thread-Safe Data ---
        with GameController.lock:
            order_queue_copy = list(GameController.order_queue)
            pickup_queue_copy = list(GameController.pickup_queue)
            ready_food_copy = dict(GameController.ready_food)
            log_copy = list(GameController.log)
            strikes_copy = GameController.strikes
            points_copy = GameController.points 

        # --- Draw Queues and Food (with timer bars) ---
        draw_text(screen, "Order Queue", title_font, (50, 20), COLOR_TITLE)
        y_offset = 70
        for qustomer in order_queue_copy:
            party = "party of 2" if qustomer.entangled else "party of 1"
            draw_text(screen, f"Qustomer #{qustomer.id} ({party})", main_font, (50, y_offset), COLOR_TEXT)
            
            bar_y = y_offset + 30
            if qustomer.timer_active:
                elapsed = time.time() - qustomer.timer_start_time
                percent_left = 1.0 - (elapsed / qustomer.timer_duration)
                percent_left = max(0.0, min(1.0, percent_left)) # Clamp 0-1
                
                bar_current_width = int(BAR_MAX_WIDTH * percent_left)
                
                # Dynamic color
                color = COLOR_BAR_HI
                if percent_left < 0.5: color = COLOR_BAR_MED
                if percent_left < 0.2: color = COLOR_BAR_LOW
                
                # Draw background
                pygame.draw.rect(screen, COLOR_BAR_BG, (50, bar_y, BAR_MAX_WIDTH, BAR_HEIGHT))
                # Draw foreground
                pygame.draw.rect(screen, color, (50, bar_y, bar_current_width, BAR_HEIGHT))
                y_offset += 25 # Extra space for the bar
            
            y_offset += 35 # Space for next customer

        draw_text(screen, "Pickup Queue", title_font, (370, 20), COLOR_TITLE)
        y_offset = 70
        for qustomer in pickup_queue_copy:
            draw_text(screen, f"Qustomer #{qustomer.id} (Order: {qustomer.order})", main_font, (370, y_offset), COLOR_TEXT)
            
            bar_y = y_offset + 30
            if qustomer.timer_active:
                elapsed = time.time() - qustomer.timer_start_time
                percent_left = 1.0 - (elapsed / qustomer.timer_duration)
                percent_left = max(0.0, min(1.0, percent_left)) # Clamp 0-1
                
                bar_current_width = int(BAR_MAX_WIDTH * percent_left)
                
                # Dynamic color
                color = COLOR_BAR_HI
                if percent_left < 0.5: color = COLOR_BAR_MED
                if percent_left < 0.2: color = COLOR_BAR_LOW
                
                # Draw background
                pygame.draw.rect(screen, COLOR_BAR_BG, (370, bar_y, BAR_MAX_WIDTH, BAR_HEIGHT))
                # Draw foreground
                pygame.draw.rect(screen, color, (370, bar_y, bar_current_width, BAR_HEIGHT))
                y_offset += 25 # Extra space
            
            y_offset += 35 # Space for next customer

        draw_text(screen, "Ready Food", title_font, (750, 20), COLOR_TITLE)
        y_offset = 70
        for item, count in ready_food_copy.items():
            draw_text(screen, f"Dish {item}: {count} servings", main_font, (750, y_offset), COLOR_TEXT)
            y_offset += 35

        # --- Draw Score and Log (using copied variables) ---
        draw_text(screen, "Score", title_font, (1000, 20), COLOR_TITLE)
        draw_text(screen, f"Points: {points_copy}", main_font, (1000, 70), COLOR_SUCCESS)
        draw_text(screen, f"Strikes: {strikes_copy}", main_font, (1000, 110), COLOR_STRIKE)

        draw_text(screen, "Game Log", title_font, (50, 350), COLOR_TITLE)
        log_rect = pygame.Rect(50, 400, 450, 280)
        pygame.draw.rect(screen, (255, 255, 255), log_rect) # White BG
        pygame.draw.rect(screen, (0,0,0), log_rect, 2) # Black border
        y_offset = 420
        line_height = 25
        max_width = log_rect.width - 40 # Increased padding

        for message in log_copy:
            words = message.split(' ')
            line = ''
            for word in words:
                test_line = line + word + ' '
                text_width, _ = log_font.size(test_line)
                if text_width <= max_width:
                    line = test_line
                else:
                    # draw the current line and start a new one
                    draw_text(screen, line, log_font, (70, y_offset), COLOR_TEXT)
                    y_offset += line_height
                    line = word + ' '

                # stop if text goes past bottom of log
                if y_offset + line_height > log_rect.bottom:
                    line = "" # Clear line to prevent drawing it
                    break

            # draw last line of text
            if line and y_offset + line_height <= log_rect.bottom:
                draw_text(screen, line, log_font, (70, y_offset), COLOR_TEXT)
                y_offset += line_height
            
            if y_offset + line_height > log_rect.bottom:
                break

        # --- Draw Buttons and Inputs ---
        btn_take_order.draw(screen, main_font)
        input_qook.draw(screen, main_font)
        btn_qook.draw(screen, main_font)
        input_serve.draw(screen, main_font)
        btn_serve.draw(screen, main_font)
        draw_text(screen, "Dish ID", log_font, (550, 525), COLOR_TEXT) 
        draw_text(screen, "Qustomer #", log_font, (550, 595), COLOR_TEXT) 
        
        # --- Handle Game Over Screen ---
        if GameController.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((100, 100, 100, 200)) # Semi-transparent grey
            screen.blit(overlay, (0, 0))
            
            final_text = "You Win!" if points_copy >= 10 else "Game Over!"
            final_color = COLOR_SUCCESS if points_copy >= 10 else COLOR_STRIKE
            
            text_surf = title_font.render(final_text, True, final_color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            pygame.draw.rect(screen, (255,255,255), text_rect.inflate(40, 40))
            screen.blit(text_surf, text_rect)

        # --- Update Display ---
        pygame.display.flip()
        clock.tick(30) # 30 FPS

    pygame.quit()
    sys.exit()

# Run the game
if __name__ == "__main__":
    run_game()