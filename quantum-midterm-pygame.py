import pygame
import sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
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
        if not cls.debug_on:
            timer_thread = threading.Thread(target=qustomer.start_timer, args=(seconds, qustomer.status))
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
        cls.set_timer(qustomer, 30) # Increased time for GUI
        
        # removed interference for now
        # if random.randint(0, 1) > 0.2:
        #     cls.log_message("A rat interfered with the order!")
        #     theta = random.uniform(0, math.pi)
        #     for i in range(1, qustomer.n + 1):
        #         qustomer.qc.p(theta, 0)
    
    @classmethod
    def valid_dish(cls, menu_item):
        if len(menu_item) != 3 or any(c not in '01' for c in menu_item):
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
        # n = 3
        # oracle = QuantumCircuit(n)
        # oracle.ccz(0, 1, 2)

        # # --- Define subset
        # subset = []
        # print("keys: " + str(cls.ready_food.keys()))
        # for key in cls.ready_food.keys():
        #     subset.append(int(key, 2))
        # amps = np.zeros(2**n, dtype=complex)
        # amps[subset] = 1/np.sqrt(len(subset))
        # amps = amps / np.linalg.norm(amps)
        # subset_bitstring = []
        # for i in subset:
        #     subset_bitstring.append(bin(i)[2:].zfill(n))
        # print(subset_bitstring)

        # psi = Statevector(amps)
        # U = cls.state_to_unitary(amps)
        # Um = QuantumCircuit(n)
        # Um.append(UnitaryGate(U), range(n))

        # problem = AmplificationProblem(
        #     oracle, state_preparation=Um, is_good_state=(subset_bitstring if len(subset_bitstring) else True)
        # )

        # grover = Grover(sampler=StatevectorSampler())
        # result = grover.amplify(problem)

        # print("Top measurement:", result.top_measurement)
        # return result.top_measurement

        # --- 1. Define the Problem ---
        n = 3
        all_states = 2**n

        # --- A. Define the Target (Needle) ---
        # The *specific item* we want to find
        target_state = '100'
        M = 1

        # --- B. Define the Search Space (Haystack) ---
        # The subset we are searching *within*
        subset_bitstrings = ['000', '001', '010', '011', '100', '101', '110']
        k = len(subset_bitstrings)

        # # Sanity check: our target MUST be in our subset
        # if target_state not in subset_bitstrings:
        #     raise ValueError("The target item is not in the search space subset.")

        print(f"Search Space (k): {k} items {subset_bitstrings}")
        print(f"Target (M): {M} item '{target_state}'\n")


        # --- 2. Build the Oracle (U_f) ---
        # This circuit marks the *target* ('101')
        oracle = QuantumCircuit(n, name="U_f (marks 101)")
        mcz = ZGate().control(n - 1)

        # '101' -> q2=1, q1=0, q0=1. Flip the '0' on q1
        target_reversed = target_state[::-1]
        for i in range(n):
            if target_reversed[i] == '0':
                oracle.x(i)
                
        oracle.append(mcz, list(range(n)))

        for i in range(n):
            if target_reversed[i] == '0':
                oracle.x(i)
                
        # print("Oracle Circuit:")
        # print(oracle.draw())


        # --- 3. Build the State Preparation (A or Um) ---
        # This circuit prepares the *subset* (haystack)
        # We use the reliable method from our previous conversation

        # Create the target statevector for the k=7 subset
        target_vector = np.zeros(all_states)
        for bitstring in subset_bitstrings:
            index = int(bitstring, 2)
            target_vector[index] = 1 / np.sqrt(k)

        # Create the circuit Um with the initialize instruction
        Um_init = QuantumCircuit(n)
        Um_init.initialize(target_vector, range(n))

        # Decompose and remove the non-unitary 'reset' gate
        Um_decomposed = Um_init.decompose()
        Um = QuantumCircuit(n, name="Um (prepares k=7)")
        for instruction in Um_decomposed.data:
            if instruction.operation.name != 'reset':
                Um.append(instruction)

        # print("\nState Prep Circuit (Um):")
        # print(Um.draw())
                
        # --- 4. Calculate Iterations & Build Problem ---
        # R = floor( (pi/4) * sqrt(k/M) )
        R = int(np.floor((np.pi / 4) * np.sqrt(k / M)))
        print(f"Optimal Iterations (R): {R}\n")

        # We provide the oracle (for the needle) and
        # the state_preparation (for the haystack)
        problem = AmplificationProblem(
            oracle=oracle,
            state_preparation=Um,
            is_good_state=None  # We use the oracle, so this isn't needed
        )

        # --- 5. Run Grover ---
        # We MUST tell Grover how many iterations to run
        grover = Grover(
            sampler=StatevectorSampler(),
            iterations=R  # Use our calculated 2 iterations
        )
        result = grover.amplify(problem)

        # --- 6. Show Results ---
        print(f"Top measurement: {result.top_measurement}")

        # Plot the results
        formatted_counts = {}
        for bitstring, prob in result.circuit_results[0].items():
            formatted_counts[bitstring] = prob

        plot_histogram(formatted_counts, title="Found '101' in 7-item Subset")

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

        # Run the search
        selected_food = cls.search_ready_subset(qustomer_to_serve)

        if selected_food is None:
            cls.log_message("No food is ready to serve!")
            # Put customer back in queue
            with cls.lock:
                cls.pickup_queue.append(qustomer_to_serve)
            return False

        # Consume the food item and 
        with cls.lock:
            cls.ready_food[selected_food] -= 1
            
            if cls.ready_food[selected_food] == 0:
                del cls.ready_food[selected_food]

        cls.log_message(f"Served {selected_food} to qustomer #{qustomer_id} (wanted {qustomer_to_serve.order})")

        # with cls.lock:
        if str(selected_food) == str(qustomer_to_serve.order) or "surprise me" in qustomer_to_serve.order:
            cls.log_message("Order served correctly!")
            cls.points += 1
            if cls.points >= 10:
                cls.log_message("Congratulations! You have successfully served 10 correct orders and won the game!")
                cls.game_over = True
        else:
            cls.log_message("Order served incorrectly </3")
            cls.points -= 1
            cls.strikes += 1
            if cls.strikes >= 3:
                cls.log_message("Game over! You have made 3 incorrect orders.")
                cls.game_over = True
        
        qustomer_to_serve.status = QustomerStatus.COMPLETE
        return True
    
    @classmethod
    def handle_inline_fail(cls, qustomer):
        try:
            with cls.lock:
                cls.order_queue.remove(qustomer)
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
            for i in range(n):
                self.order += str(np.random.randint(0, 2))
        else:
            self.order = "surprise me <( • _ • )>"
        
        with GameController.lock:
            self.id = GameController.cur_qustomer_id
            GameController.cur_qustomer_id += 1
        
        self.status = QustomerStatus.IN_LINE
        if entangled:
            self.qc = QuantumCircuit(2 * n + 1, n)
            self.maximal = random.randint(0, 1) > 0.5 # maximally or minimally entangles a pair of qustomers
        else:
            self.qc = QuantumCircuit(n + 1, n)
        
        GameController.log_message(f"Qustomer #{self.id} created (Order: {self.order})")
        GameController.qustomer_enter(self, entangled)
    
    def start_timer(self, seconds, status):
        start_time = time.time()
        while time.time() - start_time < seconds:
            if self.status != status:
                # Status changed (e.g., served), so timer is void
                return
            time.sleep(0.1)
        
        # Time's up
        if self.status == QustomerStatus.IN_LINE and status == QustomerStatus.IN_LINE:
            GameController.handle_inline_fail(qustomer = self)
        elif self.status == QustomerStatus.WAITING and status == QustomerStatus.WAITING:
            GameController.handle_waiting_fail(qustomer = self)

# --- Pygame GUI Code ---

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
            points_copy = GameController.points # needs to update after each serve

        # --- Draw Queues and Food ---
        draw_text(screen, "Order Queue", title_font, (50, 20), COLOR_TITLE)
        y_offset = 70
        for qustomer in order_queue_copy:
            party = "party of 2" if qustomer.entangled else "party of 1"
            draw_text(screen, f"Qustomer #{qustomer.id} ({party})", main_font, (50, y_offset), COLOR_TEXT)
            y_offset += 35

        draw_text(screen, "Pickup Queue", title_font, (370, 20), COLOR_TITLE)
        y_offset = 70
        for qustomer in pickup_queue_copy:
            draw_text(screen, f"Qustomer #{qustomer.id} (Order: {qustomer.order})", main_font, (370, y_offset), COLOR_TEXT)
            y_offset += 35

        draw_text(screen, "Ready Food", title_font, (750, 20), COLOR_TITLE)
        y_offset = 70
        for item, count in ready_food_copy.items():
            draw_text(screen, f"Dish {item}: {count} servings", main_font, (750, y_offset), COLOR_TEXT)
            y_offset += 35

        # --- Draw Score and Log ---
        draw_text(screen, "Score", title_font, (1000, 20), COLOR_TITLE)
        draw_text(screen, f"Points: {GameController.points}", main_font, (1000, 70), COLOR_SUCCESS)
        draw_text(screen, f"Strikes: {GameController.strikes}", main_font, (1000, 110), COLOR_STRIKE)

        draw_text(screen, "Game Log", title_font, (50, 350), COLOR_TITLE)
        log_rect = pygame.Rect(50, 400, 450, 280)
        pygame.draw.rect(screen, (255, 255, 255), log_rect) # White BG
        pygame.draw.rect(screen, (0,0,0), log_rect, 2) # Black border
        y_offset = 420
        line_height = 25
        max_width = log_rect.width - 10 # padding

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
                    break

            # draw last line of text
            if y_offset + line_height <= log_rect.bottom:
                draw_text(screen, line, log_font, (70, y_offset), COLOR_TEXT)
                y_offset += line_height
            
            # stop if log height is exceeded
            if y_offset + line_height > log_rect.bottom:
                break

        # --- Draw Buttons and Inputs ---
        btn_take_order.draw(screen, main_font)
        input_qook.draw(screen, main_font)
        btn_qook.draw(screen, main_font)
        input_serve.draw(screen, main_font)
        btn_serve.draw(screen, main_font)
        draw_text(screen, "Dish ID", log_font, (1000, 490), COLOR_TEXT)
        draw_text(screen, "Qustomer #", log_font, (1000, 560), COLOR_TEXT)
        
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