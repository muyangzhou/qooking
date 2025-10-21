from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import numpy as np
from qiskit.visualization import plot_histogram
import time
from enum import Enum
import threading
import random
import math

class QustomerStatus(Enum):
    IN_LINE = 0
    WAITING = 1
    COMPLETE = 2

class GameController:
    debug_on = True
    order_queue = []
    pickup_queue = []
    ready_food = {}
    cur_qustomer_id = 1
    strikes = 0
    successes = 0
    
    @classmethod
    def show_order_queue(cls):
        print("--- customers in line:")
        if not cls.order_queue:
            print("No orders in the queue.")
            return
        for i, qustomer in enumerate(cls.order_queue):
            print(str(i + 1) + ". qustomer #" + str(qustomer.id))

    @classmethod
    def show_pickup_queue(cls):
        print("--- customers waiting for order pickup:")
        if not cls.pickup_queue:
            print("No orders in the pickup queue.")
            return
        for i, qustomer in enumerate(cls.pickup_queue):
            print(str(i + 1) + ". qustomer #" + str(qustomer.id) + " order: " + qustomer.order)
    
    @classmethod
    def show_ready_food(cls):
        print("--- ready food:")
        if len(cls.ready_food) == 0:
            print("No food is ready.")
            return
        i = 0
        for key, value in cls.ready_food.items():
            i += 1
            print(str(i + 1) + ". menu item " + str(key) + ": " + str(value) + " servings")
    
    @classmethod
    def show_game_state(cls):
        print("\n\nGAME STATE")
        cls.show_order_queue()
        cls.show_pickup_queue()
        cls.show_ready_food()

    @classmethod
    def set_timer(cls, qustomer, seconds):
        if not cls.debug_on:
            timer_thread = threading.Thread(target=qustomer.start_timer, args=(seconds, qustomer.status))
            timer_thread.daemon = True # allows main program to exit even if thread is running
            timer_thread.start()

    @classmethod
    def qustomer_enter(cls, qustomer, entangled):
        print("Enter qustomer #" + str(qustomer.id))
        cls.order_queue.append(qustomer)
        cls.set_timer(qustomer, 6)
        for i in range(1, qustomer.n + 1):
            qustomer.qc.h(i) # put customer order into superposition
        if entangled:
            if not qustomer.maximal:
                # minimal bell state
                for i in range(1, qustomer.n + 1):
                    qustomer.qc.x(n + i)
            for i in range(1, qustomer.n + 1):
                qustomer.qc.cx(i, n + i)
            print("Created " + ("maximally" if qustomer.maximal else "minimally") + " entangled qustomer #" + str(qustomer.id) + ".5")
                    
    @classmethod
    def take_order(cls):
        qustomer = cls.order_queue.pop(0)
        print("Order " + qustomer.order + " for qustomer #" + str(qustomer.id) + " taken.")
        cls.pickup_queue.append(qustomer)
        qustomer.status = QustomerStatus.WAITING
        cls.set_timer(qustomer, 15)
        # interference may occur at this time
        if random.randint(0, 1) > 0.2:
            print("A rat was making noise in the background and interfered with your ability to take the right order!")
            theta = random.uniform(0, math.pi) # interference amount is randomly generated
            # should we make it part of the game for the player to infer if there is interference?
            for i in range(1, qustomer.n + 1):
                qustomer.qc.p(theta, 0)
            # this could be useful:
            # print("Order for qustomer #" + str(qustomer.id) + " has been altered to " + qustomer.qc.measure_all().reverse_bits().to_dict()['
    
    @classmethod
    def valid_dish(cls, menu_item):
        if len(menu_item) != 3 or any(c not in '01' for c in menu_item):
            return False
        return True
    
    @classmethod
    def prepare_dish(cls, menu_item):
        if not cls.valid_dish(menu_item):
            return False
        if not menu_item in cls.ready_food:
            # add item to ready food
            cls.ready_food[menu_item] = 0
        # increment dish count
        cls.ready_food[menu_item] += 1
        print("Prepared dish:", menu_item)
        return True
    
    @classmethod
    def search_ready(cls, qustomer): # Grover's search        
        # Unitary Oracle
        order = qustomer.order
        qc = qustomer.qc
        t = qustomer.n + 1 # total qubits, including ancilla
        if order == "surprise me":
            m = len(cls.ready_food.keys())
            # do this
            
        else:
            a = [int(bit) for bit in order] # target item as list of bits
            U = QuantumCircuit(t)

            for i in range(1,t):
                if a[i-1] == 0:
                    U.x(i)

            U.mcx(list(range(1, t)), 0, mode="noancilla")

            for i in range(1,t): # repeats n times
                if a[i-1] == 0: # to check all indexes of a from 0 to n (not included) and not just from 1 to n
                    U.x(i)

            # Diffusion Operator W
            W = QuantumCircuit(t) # acts on input qubits only

            for i in range(1, t):
                W.h(i)

            for i in range(1, t):
                W.x(i)

            W.mcp(np.pi,list(range(1,t-1)),n)

            for i in range(1, t):
                W.x(i)

            for i in range(1, t):
                W.h(i)

            R = int(np.floor(np.pi*np.sqrt(2**n)/4)) # N is 2^n

            qc = QuantumCircuit(t,n)
            qc.x(0)
            qc.h(0)
            qc.barrier()

            qc.h(range(1,t))
            qc.barrier()

            for i in range(R):
                qc.compose(U,qubits=(list(range(t))),inplace=True)
                qc.barrier()
                qc.compose(W,qubits=(list(range(t))),inplace=True)
                qc.barrier()

            qc.h(0)
            qc.x(0)
            qc.barrier()
            for i in range(1,t):
                qc.measure(i, i - 1)
        

        backend = Aer.get_backend('qasm_simulator')
        counts= backend.run(qc, shots=1024).result().get_counts(qc)

        most_frequent_outcome = max(counts, key=counts.get)

        big_endian = most_frequent_outcome[::-1]
        return str(big_endian)

        # ingredients = {
        #     "000": ["apple", "flour", "sugar", "cinnamon", "butter"],
        #     "001": ["pumpkin", "flour", "sugar", "eggs", "cinnamon"],
        #     "011": ["coffee beans", "water"],
        #     "100": ["tea leaves", "milk", "sugar", "cinnamon"],
        #     "101": ["pumpkin", "milk", "coffee beans", "sugar", "cinnamon"],
        #     "110": ["flour", "sugar", "cinnamon", "butter", "eggs"],
        #     "111": ["blueberry", "flour", "sugar", "eggs", "milk"],
        #     "010": ["banana", "flour", "sugar", "eggs", "butter"]
        # }

        # for i in menu:
        #     if i == bin(int(big_endian[1:])):
        #         return menu[i]
    
    @classmethod
    def serve_food(cls, qustomer_index):
        if not qustomer_index.isdigit() or not 0 <= int(qustomer_index) < cls.cur_qustomer_id:
            print("Invalid qustomer index.")
            return False
        for i, qustomer in enumerate(cls.pickup_queue):
            if qustomer.id == int(qustomer_index):
                # if qustomer.order not in cls.ready_food:
                #     print("this dish is not ready!")
                #     return False
                selected_food = cls.search_ready(qustomer)
                menu = ["apple pie", "pumpkin pie", "espresso", "chai latte", "pumpkin spice latte", 
                    "cinnamon roll", "blueberry muffin", "banana bread"]
                # print("Putting together the ingredients: " + ", ".join(ingredients))
                cls.pickup_queue.pop(i)
                cls.ready_food[selected_food] -= 1
                if cls.ready_food[selected_food] == 0:
                    del cls.ready_food[selected_food]
                print("served order " + selected_food + " to qustomer #" + str(qustomer.id) + " who wanted " + qustomer.order)
                if str(selected_food) == str(qustomer.order) or qustomer.order == "surprise me":
                    print("Order served correctly!")
                    cls.successes += 1
                    if cls.successes >= 10:
                        print("Congratulations! You have successfully served 10 correct orders and won the game!")
                        exit()
                else:
                    print("Order served incorrectly </3")
                    cls.strikes += 1
                    if cls.strikes >= 3:
                        print("Game over! You have made 3 incorrect orders.")
                        exit()
                qustomer.status = QustomerStatus.COMPLETE
                return True
        print("No qustomer with this order in the pickup queue.")
        return False
    
    @classmethod
    def handle_inline_fail(cls, qustomer):
        cls.order_queue.remove(qustomer)
        print("customer left before ordering. press enter to continue")
    
    @classmethod
    def handle_waiting_fail(cls, qustomer):
        cls.pickup_queue.remove(qustomer)
        print("customer left before food was served. press enter to continue")
    
    @classmethod
    def spawn_qustomer(cls, n, min_interval, max_interval):
        while True:
            qustomer = Qustomer(n)
            time.sleep(np.random.randint(min_interval, max_interval))
            

class Qustomer:
    def __init__(self, n, entangled=False, surprise_me=False):
        if entangled:
            print("hi")
        self.n = n
        self.order = ""
        if not surprise_me:
            for i in range(n):
                self.order += str(np.random.randint(0, 2))
        else:
            self.order = "surprise me"
        self.id = GameController.cur_qustomer_id
        GameController.cur_qustomer_id += 1
        self.status = QustomerStatus.IN_LINE
        if entangled:
            self.qc = QuantumCircuit(2 * n + 1, n)
            self.maximal = random.randint(0, 1) > 0.5 # random bool
        else:
            self.qc = QuantumCircuit(n + 1, n)
        print("qustomer created with", n, "qubits.")
        GameController.qustomer_enter(self, entangled)  # Use class method
        # print("qustomer created with", n, "qubits.")
        # GameController.qustomer_enter(qustomer=self)  # Use class method
        # self.start_timer(6, QustomerStatus.IN_LINE) # 6 seconds in line
    
    def start_timer(self, seconds, status):
        time.sleep(seconds)
        # time's up
        if self.status == QustomerStatus.IN_LINE and status == QustomerStatus.IN_LINE:
            GameController.handle_inline_fail(qustomer = self)
        elif self.status == QustomerStatus.WAITING and status == QustomerStatus.WAITING:
            GameController.handle_waiting_fail(qustomer = self)
        # else success
    
    def draw(self, filename):
        self.qc.draw("mpl", filename="_midterm/" + filename)
        
if __name__ == "__main__":
    print("program start")
    n = 3 # 2^n menu items
    qustomer = Qustomer(n)
    qustomer = Qustomer(n, True)
    qustomer = Qustomer(n, True, True)
    if not GameController.debug_on:
        spawn_thread = threading.Thread(target=GameController.spawn_qustomer, args=(n, 2, 5))
        spawn_thread.daemon = True
        spawn_thread.start()

    # qustomer.draw("quantum-midterm-circuit.png")
    while True:
        GameController.show_game_state()
        choice = input("what do you want to do? ")
        if choice == "take order":
            GameController.take_order()
        elif choice == "qook":
            menu_item = input("which menu item (binary str, len = 3)? ")
            # runs the cooking algorithm, divided into steps: searching for ingredients, putting ingredients together (simple)
            while not GameController.prepare_dish(menu_item):
                menu_item = input("Invalid menu item. which menu item? ")
        elif choice == "serve":
            order_index = input("which qustomer to serve? ")
            while not GameController.serve_food(order_index):
                order_index = input("Failed to serve this qustomer. which qustomer to serve? ")
            print("successfully served qustomer #" + order_index)

