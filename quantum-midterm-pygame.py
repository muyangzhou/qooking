import pygame
import sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer, AerSimulator
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
from qiskit.quantum_info import Statevector, DensityMatrix
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

    BITSTRING_TO_ITEM = {
        "000": "apple pie",
        "001": "pumpkin pie",
        "010": "banana bread",
        "011": "coffee",
        "100": "chai latte",
        "101": "pumpkin spice latte",
        "110": "cinnamon roll"
    }
    recipes_key = {
             "000": ["apple", "flour", "sugar", "cinnamon", "butter"],
             "001": ["pumpkin", "flour", "sugar", "eggs", "cinnamon"],
             "011": ["coffee", "milk"],
             "100": ["chai", "milk", "sugar", "cinnamon"],
             "101": ["pumpkin", "milk", "coffee beans", "sugar", "cinnamon"],
             "110": ["flour", "sugar", "cinnamon", "butter", "eggs"],
             "010": ["banana", "flour", "sugar", "eggs", "butter"]
             } # apple pie, pumpkin pie, coffee, chai latte, pumpkin spice latte, cinnamon roll, banana bread
    
    ingredients_map = { "apple": lambda: (lambda qc, q: qc.rz(np.pi/3, q)), 
                           "pumpkin": lambda: (lambda qc, q: qc.rx(np.pi/4, q)),
                           "cinnamon": lambda: (lambda qc, q: qc.ry(np.pi/6, q)),
                           "coffee": lambda: (lambda qc, q: qc.p(np.pi, q)),
                           "chai": lambda: (lambda qc, q: qc.p(np.pi/2, q)),
                           "banana": lambda: (lambda qc, q: qc.p(np.pi/4, q)), 
                           "sugar": lambda: (lambda qc, q: qc.cx(1, 2)),
                           "milk": lambda: (lambda qc, q: qc.cz(2, 1)), 
                           "eggs": lambda: (lambda qc, q: qc.h(q)), 
                           "butter": lambda: (lambda qc, q: qc.rx(np.pi/6, q)),
                           "flour": lambda: (lambda qc, q: qc.rz(np.pi/4, q))
                        }
    
    expected_density_matrices = {}


    log = ["Welcome to the Quantum Qafe!"] # GUI will display this
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

    # timer thread
    @classmethod
    def set_timer(cls, qustomer, seconds):
        qustomer.timer_start_time = time.time()
        qustomer.timer_duration = seconds
        qustomer.timer_active = True
        
        with cls.lock:
            qustomer.timer_id += 1 # Invalidate any old timer threads
            current_timer_id = qustomer.timer_id
        
        if not cls.debug_on:
            timer_thread = threading.Thread(target=qustomer.start_timer, args=(seconds, qustomer.status, current_timer_id))
            timer_thread.daemon = True
            timer_thread.start()

    # new qustomer enters cafe
    @classmethod
    def qustomer_enter(cls, qustomer):
        cls.log_message(f"Enter qustomer #{qustomer.id}")
        with cls.lock:
            cls.order_queue.append(qustomer)
        cls.set_timer(qustomer, 60) # TODO change this
            
    # user wants to take order for next qustomer in line
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
        
        cls.log_message(f"Order for qustomer #{qustomer.id} taken. (In Superposition)")
        qustomer.status = QustomerStatus.WAITING
        cls.set_timer(qustomer, 60) # TODO change this
    
    # check if menu_item is valid
    @classmethod
    def valid_dish(cls, menu_item):
        if len(menu_item) != 3 or any(c not in '01' for c in menu_item) or menu_item == '111':
            return False
        return True
    
    # user wants to qook menu_item
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
    
    # Convert state vector to reflection unitary
    @classmethod
    def state_to_unitary(cls, psi):
        n = int(np.log2(len(psi)))
        zero = np.zeros_like(psi)
        zero[0] = 1
        v = psi - zero
        v = v / np.linalg.norm(v)
        H = np.eye(2**n) - 2 * np.outer(v, np.conj(v))
        return H

    # search ready food subset for qustomer's wanted order (or random order if surprise me)
    @classmethod
    def search_ready_subset(cls, qustomer, wanted_order): # Argument added
        n = qustomer.n
        all_states = 2**n

        # target start (qustomer order)
        # set this to 111 if qustomer has "surprise me" order (grover's algo with result in uniform superposition)
        target_state = wanted_order if "surprise me" not in wanted_order else '111'
        M = 1

        # ready food
        with cls.lock: # Added lock for thread safety
            subset_bitstrings = list(cls.ready_food.keys())
            k = len(subset_bitstrings)
        
        if k == 0:
            print("Search error: k=0")
            return "000" 

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
        Um = QuantumCircuit(n, name=f"Um (prepares k={k})") # Name dynamically
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
            is_good_state=None 
        )

        # run grover
        grover = Grover(
            sampler=StatevectorSampler(),
            iterations=R 
        )
        result = grover.amplify(problem) 

        # debug
        print(f"Top measurement: {result.top_measurement}")
        formatted_counts = {}
        for bitstring, prob in result.circuit_results[0].items():
            formatted_counts[bitstring] = prob
        
        return result.top_measurement

    # collapse superposition of qustomer with qustomer_index
    @classmethod
    def measure_order(cls, qustomer_index):
        BITSTRING_TO_ITEM = {
        "000": "apple pie",
        "001": "pumpkin pie",
        "010": "banana bread",
        "011": "coffee",
        "100": "chai latte",
        "101": "pumpkin spice latte",
        "110": "cinnamon roll"
    }
        if not qustomer_index.isdigit():
            cls.log_message(f"Invalid qustomer index: {qustomer_index}. Must be a number.")
            return False
        
        qustomer_id = int(qustomer_index)
        qustomer_to_measure = None
        
        with cls.lock:
            for qustomer in cls.pickup_queue:
                if qustomer.id == qustomer_id:
                    qustomer_to_measure = qustomer
                    break
        
        if not qustomer_to_measure:
            cls.log_message(f"No qustomer with ID #{qustomer_id} in pickup queue.")
            return False
            
        if qustomer_to_measure.is_collapsed:
            cls.log_message(f"Qustomer #{qustomer_id}'s order is already collapsed.")
            return True

        # --- DUMMY GAME PLACEHOLDER ---
        cls.log_message(f"Starting measurement game for Qustomer #{qustomer_id}...")
        time.sleep(1) # Simulating the dummy game
        cls.log_message("...Measurement game complete!")
        # --- END DUMMY GAME ---
        
        with cls.lock:
            # Case 1: Entangled customer
            if qustomer_to_measure.entangled_partner:
                partner = qustomer_to_measure.entangled_partner
                
                # Check if partner was measured first
                if partner.is_collapsed:
                    # This is the second partner. Just reveal their order.
                    qustomer_to_measure.is_collapsed = True
                    qustomer_to_measure.order_revealed = True
                    cls.log_message(f"Partner was already measured! Qustomer #{qustomer_id}'s order is revealed: {qustomer_to_measure.collapsed_order}")
                else:
                    # This is the first partner. Measure the pair.
                    cls.log_message(f"Measuring entangled pair: #{qustomer_to_measure.id} and #{partner.id}")
                    
                    qc = qustomer_to_measure.qc
                    n = qustomer_to_measure.n
                    qc.measure(list(range(1, 2 * n + 1)), list(range(0, 2 * n)))
                    
                    backend = Aer.get_backend('qasm_simulator')
                    while True:
                        counts = backend.run(qc, shots=1).result().get_counts(qc)
                        measurement = list(counts.keys())[0]
                        measurement_big_endian = measurement[::-1]
                        if measurement_big_endian[0:n] != '111' and measurement_big_endian[n:2*n] != '111':
                            break
                    
                    print(measurement_big_endian)
                    order_A = measurement_big_endian[0 : n] # This customer
                    order_B = measurement_big_endian[n : 2*n] # Partner
                    
                    # Set both to collapsed
                    qustomer_to_measure.is_collapsed = True
                    partner.is_collapsed = True
                    
                    # Store collapsed orders
                    qustomer_to_measure.collapsed_order = order_A
                    qustomer_to_measure.item_name = BITSTRING_TO_ITEM.get(order_A, "unknown item")
                    cls.log_message("Item name assigned:", qustomer_to_measure.item_name)

                    partner.collapsed_order = order_B
                    partner.item_name = BITSTRING_TO_ITEM.get(order_B, "unknown item")
                    cls.log_message("Item name assigned:", partner.item_name)

                    # --- REQUEST 1 LOGIC ---
                    # Reveal *this* customer's order
                    qustomer_to_measure.order_revealed = True
                    # Keep partner's order *hidden*
                    partner.order_revealed = False 
                    
                    cls.log_message(f"Collapse! Qustomer #{qustomer_id}'s order is {order_A}.")
                    cls.log_message(f"Qustomer #{partner.id}'s order is now also set (but hidden)!")
            
            # Case 2: "Surprise Me" customer
            elif "surprise me" in qustomer_to_measure.order:
                # Pick a random 3-bit string that isn't '111'
                new_order = ""
                while True:
                    new_order = ""
                    for i in range(qustomer_to_measure.n):
                        new_order += str(np.random.randint(0, 2))
                    if new_order != "111":
                        break
                
                qustomer_to_measure.order = new_order # Overwrite "surprise me"
                qustomer_to_measure.is_collapsed = True
                qustomer_to_measure.order_revealed = True
                cls.log_message(f"Qustomer #{qustomer_id} decided they want: {new_order}")

            # Case 3: Standard customer
            else:
                qustomer_to_measure.is_collapsed = True
                qustomer_to_measure.order_revealed = True
                cls.log_message(f"Measurement complete for Qustomer #{qustomer_id}. Order confirmed: {qustomer_to_measure.order}")
        
        return True


    @classmethod
    def prepare_recipe_state(cls, qustomer_to_measure, recipes_key, ingredients_map):
        # 1) From the binary bitstring order, identify the menu item from recipes dict
        # 2) Represent each recipe as the (initial/expected) quantum state
        bitstring = qustomer_to_measure.collapsed_order
        n = 3
        qc = QuantumCircuit(n)

        # for loop that indexes the keys like 000, 001, 010, etc.
        if bitstring in recipes_key:
            recipe_ingredients = recipes_key[bitstring]
            qubit_index = 0

            for ingredient in recipe_ingredients:
                if ingredient in ingredients_map:
                    gate_factory = ingredients_map[ingredient]
                    gate_fn = gate_factory()
                    # apply gates on consecutive qubits
                    # and two-qubit gates use the hardcoded indices from ingredients_map
                    import inspect
                    sig = inspect.signature(gate_fn)
                    num_params = len(sig.parameters) - 1 # Subtracting qc parameter

                    if num_params == 1: # single qubit gate
                        gate_fn(qc, qubit_index % n)
                        qubit_index += 1
                    elif num_params == 2: # two qubit gate with hardcoded indices
                        gate_fn(qc, None)

                elif ingredient not in ingredients_map:
                    print(f"Warning: No gate defined for ingredient '{ingredient}'")
            state = Statevector.from_instruction(qc)
            rho = DensityMatrix(state)
            cls.expected_density_matrices[bitstring] = rho.data
        else:
            print(f"Error: Bitstring '{qustomer_to_measure.collapsed_order}' not found in recipes.")
            return None
        return rho.data
    
    
    # user wants to serve food to qustomer with qustomer_index
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
                    qustomer_to_serve = qustomer # Keep in queue for now
                    break
        
        if not qustomer_to_serve:
            cls.log_message(f"No qustomer with ID #{qustomer_id} in pickup queue.")
            return False
            
        # --- NEW GATE: Check for collapse ---
        if not qustomer_to_serve.is_collapsed:
            cls.log_message(f"Order for Qustomer #{qustomer_id} is not measured! Use 'Measure' first.")
            return False
        # --- END NEW GATE ---

        wanted_order = ""
        if qustomer_to_serve.entangled_partner:
            wanted_order = qustomer_to_serve.collapsed_order
        else:
            wanted_order = qustomer_to_serve.order

        # Check if food is ready
        with cls.lock:
            if not cls.ready_food:
                cls.log_message("No food is ready to serve!")
                return False

        # Run the search, passing the determined wanted_order
        selected_food = cls.search_ready_subset(qustomer_to_serve, wanted_order) 

        # Consume the food item
        with cls.lock:
            if selected_food not in cls.ready_food:
                cls.log_message(f"Grover found {selected_food}, but it's not ready!")
                return False
                
            cls.ready_food[selected_food] -= 1
            if cls.ready_food[selected_food] == 0:
                del cls.ready_food[selected_food]
            
            # Now actually remove from queue
            cls.pickup_queue.remove(qustomer_to_serve)

        cls.log_message(f"Served {selected_food} to qustomer #{qustomer_id} (wanted {wanted_order})")

        with cls.lock:
            if str(selected_food) == str(wanted_order): # "surprise me" is already replaced
                cls.log_message("Order served correctly!")
                cls.points += 1
                if cls.points >= 10:
                    cls.log_message("Congratulations! You have successfully served 10 correct orders and won the game!")
                    cls.game_over = True
            else:
                cls.log_message("Order served incorrectly </3")
                cls.points -= 1
                cls.strikes += 1
                if cls.points < 0 or cls.strikes >= 3:
                    cls.log_message("You have no more points left." if cls.points < 0 else "")
                    cls.log_message("Game over! You have made 3 incorrect orders.")
                    cls.game_over = True
        
        qustomer_to_serve.status = QustomerStatus.COMPLETE
        qustomer_to_serve.timer_active = False
        return True
    
    @classmethod
    def handle_inline_fail(cls, qustomer):
        try:
            with cls.lock:
                cls.order_queue.remove(qustomer)
            cls.log_message(f"Qustomer #{qustomer.id} left before ordering.")
        except ValueError:
            pass 

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
            if cls.game_over: break 
            
            entangled = random.choice([True, False])
            surprise = random.choice([True, False])
            threading.Thread(target=Qustomer, args=(n, entangled, surprise), daemon=True).start()
            

class Qustomer: 
    def __init__(self, n, entangled=False, surprise_me=False):
        self.n = n
        self.entangled_partner = None
        self.collapsed_order = None 
        self.qc = None
        
        self.is_collapsed = False
        self.order_revealed = False
        
        with GameController.lock:
            self.id = GameController.cur_qustomer_id
            GameController.cur_qustomer_id += 1
        
        self.status = QustomerStatus.IN_LINE
        self.timer_start_time = 0.0
        self.timer_duration = 0.0
        self.timer_active = False
        self.timer_id = 0

        if entangled:
            self.order = "Entangled" 
            if _is_secondary:
                self.qc = _primary_qustomer.qc
                self.maximal = _primary_qustomer.maximal
                self.order = ("Maximally" if _primary_qustomer.maximal else "Minimally") + " entangled w/ #" + str(_primary_qustomer.id)
            else:
                self.qc = QuantumCircuit(2 * self.n + 1, 2 * self.n) 
                self.maximal = random.randint(0, 1) > 0.5
                
                for i in range(1, self.n + 1):
                    self.qc.h(i)
                
                if not self.maximal:
                     for i in range(1, self.n + 1):
                         self.qc.x(self.n + i)
                for i in range(1, self.n + 1):
                    self.qc.cx(i, self.n + i)
                
                partner = Qustomer(n, entangled=True, _is_secondary=True, _primary_qustomer=self)
                
                self.entangled_partner = partner
                partner.entangled_partner = self
                
                self.order = ("Maximally" if self.maximal else "Minimally") + " entangled w/ #" + str(partner.id)
                GameController.log_message(f"Created entangled pair: Qustomer #{self.id} and #{partner.id}")
                
                GameController.qustomer_enter(self)
                GameController.qustomer_enter(partner)
        
        else: # Not entangled
            self.qc = QuantumCircuit(self.n + 1, self.n)
            if not surprise_me:
                while True:
                    self.order = ""
                    for i in range(n):
                        self.order += str(np.random.randint(0, 2))
                    if self.order != "111":
                        break
            else:
                self.order = "surprise me" # <( • _ • )>"
            
            for i in range(1, self.n + 1):
                self.qc.h(i)
                
            GameController.log_message(f"Qustomer #{self.id} created (Order: {self.order})")
            GameController.qustomer_enter(self) 
    
    def start_timer(self, seconds, status_when_started, timer_id):
        start_time = time.time()
        while time.time() - start_time < seconds:
            with GameController.lock:
                if self.timer_id != timer_id:
                    return 
                if self.status == QustomerStatus.COMPLETE:
                    self.timer_active = False
                    return
            time.sleep(0.1)
        
        with GameController.lock:
            if self.timer_id != timer_id or self.status == QustomerStatus.COMPLETE:
                return
        
        self.timer_active = False 
        
        if self.status == status_when_started:
            if self.status == QustomerStatus.IN_LINE:
                GameController.handle_inline_fail(qustomer = self)
            elif self.status == QustomerStatus.WAITING:
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

# pygame main loop
def run_game():
    pygame.init()
    
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Qafe")
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

    # --- MODIFIED: GUI Elements ---
    btn_take_order = Button((550, 470, 200, 50), "Take Order", (0, 150, 0))
    
    # New Measure button
    input_measure = InputBox((550, 550, 200, 50), COLOR_TEXT)
    btn_measure = Button((780, 550, 200, 50), "Measure Order", (200, 100, 0))
    
    input_qook = InputBox((550, 630, 200, 50), COLOR_TEXT)
    btn_qook = Button((780, 630, 200, 50), "Let's Qook", (150, 0, 0))
    
    # Moved Serve button
    input_serve = InputBox((550, 710, 200, 50), COLOR_TEXT)
    btn_serve = Button((780, 710, 200, 50), "Serve", (0, 0, 150))

    # backend logic
    n = 3 # 2^n menu items
    threading.Thread(target=Qustomer, args=(n, True, False), daemon=True).start()
    time.sleep(0.1) 
    threading.Thread(target=Qustomer, args=(n, False, True), daemon=True).start() # Add a "surprise me"
    
    if not GameController.debug_on:
        spawn_thread = threading.Thread(target=GameController.spawn_qustomer, args=(n, 5, 12)) 
        spawn_thread.daemon = True
        spawn_thread.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not GameController.game_over:
                input_qook.handle_event(event)
                input_measure.handle_event(event)
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
                    
                    elif btn_measure.is_clicked(event.pos):
                        idx = input_measure.text
                        if idx:
                            threading.Thread(target=GameController.measure_order, args=(idx,), daemon=True).start()
                            input_measure.text = ""
                            input_measure.active = False

                    elif btn_serve.is_clicked(event.pos):
                        idx = input_serve.text
                        if idx:
                            threading.Thread(target=GameController.serve_food, args=(idx,), daemon=True).start()
                            input_serve.text = ""
                            input_serve.active = False

        screen.fill(COLOR_BG)
        
        with GameController.lock:
            order_queue_copy = list(GameController.order_queue)
            pickup_queue_copy = list(GameController.pickup_queue)
            ready_food_copy = dict(GameController.ready_food)
            log_copy = list(GameController.log)
            strikes_copy = GameController.strikes
            points_copy = GameController.points 

        draw_text(screen, "Order Queue", title_font, (50, 20), COLOR_TITLE)
        y_offset = 70
        for qustomer in order_queue_copy:
            draw_text(screen, f"Qustomer #{qustomer.id}", main_font, (50, y_offset), COLOR_TEXT)
            
            bar_y = y_offset + 30
            if qustomer.timer_active:
                elapsed = time.time() - qustomer.timer_start_time
                percent_left = 1.0 - (elapsed / qustomer.timer_duration)
                percent_left = max(0.0, min(1.0, percent_left)) 
                
                bar_current_width = int(BAR_MAX_WIDTH * percent_left)
                color = COLOR_BAR_HI
                if percent_left < 0.5: color = COLOR_BAR_MED
                if percent_left < 0.2: color = COLOR_BAR_LOW
                
                pygame.draw.rect(screen, COLOR_BAR_BG, (50, bar_y, BAR_MAX_WIDTH, BAR_HEIGHT))
                pygame.draw.rect(screen, color, (50, bar_y, bar_current_width, BAR_HEIGHT))
                y_offset += 25 
            
            y_offset += 35 

        draw_text(screen, "Pickup Queue", title_font, (390, 20), COLOR_TITLE)
        y_offset = 70
        for qustomer in pickup_queue_copy:
            order_str = qustomer.order
            if qustomer.entangled_partner:
                if qustomer.order_revealed: 
                    item_display = getattr(qustomer, "item_name", qustomer.collapsed_order)
                    order_str = f"Collapsed to {item_display}" 
                elif qustomer.is_collapsed: # Collapsed but not revealed (2nd partner)
                    order_str = ("Maximally" if qustomer.maximal else "Minimally") + f" entangled with #{qustomer.entangled_partner.id}"
                else: # Not yet collapsed
                    order_str = qustomer.order # "Entangled w/ #X"
            elif qustomer.is_collapsed:
                 # order_str = qustomer.order
                 order_str = getattr(qustomer, "item_name", qustomer.order)
            else:
                 # Standard or Surprise Me, not yet measured
                 order_str = f"In Superposition"
            
            draw_text(screen, f"Qustomer #{qustomer.id} (Order: {order_str})", main_font, (390, y_offset), COLOR_TEXT)
            
            bar_y = y_offset + 30
            if qustomer.timer_active:
                elapsed = time.time() - qustomer.timer_start_time
                percent_left = 1.0 - (elapsed / qustomer.timer_duration)
                percent_left = max(0.0, min(1.0, percent_left))
                
                bar_current_width = int(BAR_MAX_WIDTH * percent_left)
                color = COLOR_BAR_HI
                if percent_left < 0.5: color = COLOR_BAR_MED
                if percent_left < 0.2: color = COLOR_BAR_LOW
                
                pygame.draw.rect(screen, COLOR_BAR_BG, (390, bar_y, BAR_MAX_WIDTH, BAR_HEIGHT))
                pygame.draw.rect(screen, color, (390, bar_y, bar_current_width, BAR_HEIGHT))
                y_offset += 25
            
            y_offset += 35 

        draw_text(screen, "Ready Food", title_font, (820, 20), COLOR_TITLE)
        y_offset = 70
        for item, count in ready_food_copy.items():
            draw_text(screen, f"Dish {item}: {count} servings", main_font, (820, y_offset), COLOR_TEXT)
            y_offset += 35

        # --- Draw Ingredients Legend ---
        draw_text(screen, "Ingredients", title_font, (50, 270), COLOR_TITLE)
        draw_text(screen, "Seasonal Items", main_font, (50, 310), COLOR_TEXT)
        draw_text(screen, "apple", main_font, (250, 310), COLOR_TEXT)
        draw_text(screen, "banana", main_font, (330, 310), COLOR_TEXT)
        draw_text(screen, "chai", main_font, (450, 310), COLOR_TEXT)
        draw_text(screen, "cinnamon", main_font, (530, 310), COLOR_TEXT)
        draw_text(screen, "coffee", main_font, (670, 310), COLOR_TEXT)
        draw_text(screen, "pumpkin", main_font, (760, 310), COLOR_TEXT)


        draw_text(screen, "Baking", main_font, (50, 350), COLOR_TEXT)
        draw_text(screen, "butter", main_font, (250, 350), COLOR_TEXT)
        draw_text(screen, "eggs", main_font, (330, 350), COLOR_TEXT)
        draw_text(screen, "flour", main_font, (410, 350), COLOR_TEXT)
        draw_text(screen, "milk", main_font, (490, 350), COLOR_TEXT)
        draw_text(screen, "sugar", main_font, (570, 350), COLOR_TEXT)

        # --- Draw Score and Log ---
        draw_text(screen, "Score", title_font, (1070, 20), COLOR_TITLE)
        draw_text(screen, f"Points: {points_copy}", main_font, (1070, 70), COLOR_SUCCESS)
        draw_text(screen, f"Strikes: {strikes_copy}", main_font, (1070, 110), COLOR_STRIKE)

        draw_text(screen, "Game Log", title_font, (50, 430), COLOR_TITLE)
        log_rect = pygame.Rect(50, 470, 450, 280)
        pygame.draw.rect(screen, (255, 255, 255), log_rect)
        pygame.draw.rect(screen, (0,0,0), log_rect, 2)
        y_offset = 490
        line_height = 25
        max_width = log_rect.width - 40 

        for message in log_copy:
            words = message.split(' ')
            line = ''
            for word in words:
                test_line = line + word + ' '
                text_width, _ = log_font.size(test_line)
                if text_width <= max_width:
                    line = test_line
                else:
                    draw_text(screen, line, log_font, (70, y_offset), COLOR_TEXT)
                    y_offset += line_height
                    line = word + ' '
                if y_offset + line_height > log_rect.bottom:
                    line = "" 
                    break

            if line and y_offset + line_height <= log_rect.bottom:
                draw_text(screen, line, log_font, (70, y_offset), COLOR_TEXT)
                y_offset += line_height
            
            if y_offset + line_height > log_rect.bottom:
                break

        # --- MODIFIED: Draw Buttons and Inputs ---
        btn_take_order.draw(screen, main_font)
        input_qook.draw(screen, main_font)
        btn_qook.draw(screen, main_font)
        input_measure.draw(screen, main_font) # New
        btn_measure.draw(screen, main_font)   # New
        input_serve.draw(screen, main_font)
        btn_serve.draw(screen, main_font)
        
        draw_text(screen, "Qustomer #", log_font, (550, 605), COLOR_TEXT) # Label for measure
        draw_text(screen, "Dish ID", log_font, (550, 685), COLOR_TEXT) 
        draw_text(screen, "Qustomer #", log_font, (550, 765), COLOR_TEXT) # Label for serve
        # --- END MODIFIED ---
        
        # --- Handle Game Over Screen ---
        if GameController.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((100, 100, 100, 200)) 
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