from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import numpy as np
from qiskit.visualization import plot_histogram

class GameController:
    order_queue = []
    pickup_queue = []
    ready_food = []
    cur_qustomer_id = 1
    
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
        if not cls.ready_food:
            print("No food is ready.")
            return
        for i, item in enumerate(cls.ready_food):
            print(str(i + 1) + ". menu item " + item)
    
    @classmethod
    def show_game_state(cls):
        print("\n\nGAME STATE")
        cls.show_order_queue()
        cls.show_pickup_queue()
        cls.show_ready_food()

    @classmethod
    def qustomer_enter(cls, qustomer):
        print("Enter qustomer #" + str(qustomer.id))
        cls.order_queue.append(qustomer)
    
    @classmethod
    def take_order(cls):
        qustomer = cls.order_queue.pop(0)
        print("Order " + qustomer.order + " for qustomer #" + str(qustomer.id) + " taken.")
        cls.pickup_queue.append(qustomer)
    
    @classmethod
    def valid_dish(cls, menu_item):
        if len(menu_item) != 3 or any(c not in '01' for c in menu_item):
            return False
        return True
    
    @classmethod
    def prepare_dish(cls, menu_item):
        if not cls.valid_dish(menu_item):
            return False
        cls.ready_food.append(menu_item)
        print("Prepared dish:", menu_item)
        return True
    
    @classmethod
    def serve_food(cls, qustomer_index):
        if not qustomer_index.isdigit() or not 0 <= int(qustomer_index) < cls.cur_qustomer_id:
            print("Invalid qustomer index.")
            return False
        for i, qustomer in enumerate(cls.pickup_queue):
            if qustomer.id == int(qustomer_index):
                if qustomer.order not in cls.ready_food:
                    print("this dish is not ready!")
                    return False
                cls.pickup_queue.pop(i)
                cls.ready_food.remove(qustomer.order)
                print("served order " + qustomer.order + " to qustomer #" + str(qustomer.id))
                return True
        print("No qustomer with this order in the pickup queue.")
        return False

class Qustomer:
    def __init__(self, n, order):
        self.n = n
        self.order = order
        self.id = GameController.cur_qustomer_id
        GameController.cur_qustomer_id += 1
        self.qc = QuantumCircuit(n, 1)
        print("qustomer created with", n, "qubits.")
        GameController.qustomer_enter(qustomer=self)  # Use class method
        
    def draw(self, filename):
        self.qc.draw("mpl", filename="_midterm/" + filename)
        
if __name__ == "__main__":
    print("program start")
    n = 3 # 2^n menu items
    order = ""
    for i in range(n):
        order += str(np.random.randint(0, 2))
    qustomer = Qustomer(n, order)
    # qustomer.draw("quantum-midterm-circuit.png")
    while True:
        GameController.show_game_state()
        choice = input("what do you want to do? ")
        if choice == "take order":
            GameController.take_order()
        elif choice == "cook":
            menu_item = input("which menu item (binary str, len = 3)? ")
            while not GameController.prepare_dish(menu_item):
                menu_item = input("Invalid menu item. which menu item? ")
        elif choice == "serve":
            order_index = input("which qustomer to serve? ")
            while not GameController.serve_food(order_index):
                order_index = input("Failed to serve this qustomer. which qustomer to serve? ")
            print("successfully served qustomer #" + order_index)

