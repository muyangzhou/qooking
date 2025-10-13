import pygame
import random
import time
import sys

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
GRAY = (200, 200, 200)
LIGHT_GREEN = (144, 238, 144)
LIGHT_RED = (255, 182, 193)

# Game settings
FPS = 60
MAX_FAILED_ORDERS = 5
ORDER_TIME = 25  # Time to take order
PREP_TIME = 30   # Time to prepare order

# Food items
FOOD_ITEMS = [
    "Burger", "Pizza", "Pasta", "Salad", "Soup", 
    "Steak", "Tacos", "Sushi", "Ramen", "Sandwich",
    "Curry", "Stir Fry", "BBQ", "Seafood", "Dessert"
]

class Customer:
    def __init__(self, customer_id):
        self.id = customer_id
        self.desired_food = random.choice(FOOD_ITEMS)
        self.current_phase = "waiting_to_order"
        self.time_remaining = ORDER_TIME
        self.start_time = time.time()
        self.player_prepared_food = None
        self.position = 0
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
    def update_timer(self):
        elapsed = time.time() - self.start_time
        if self.current_phase == "waiting_to_order":
            self.time_remaining = max(0, ORDER_TIME - elapsed)
        else:  # waiting_to_pickup
            self.time_remaining = max(0, PREP_TIME - elapsed)
        return self.time_remaining <= 0
    
    def take_order(self):
        """Move customer from waiting to order to waiting to pickup"""
        if self.current_phase == "waiting_to_order":
            self.current_phase = "waiting_to_pickup"
            self.time_remaining = PREP_TIME
            self.start_time = time.time()
            return True
        return False
    
    def fulfill_order(self, prepared_food):
        """Player fulfills the order with prepared food"""
        if self.current_phase == "waiting_to_pickup":
            self.player_prepared_food = prepared_food
            return self.evaluate_order()
        return False
    
    def evaluate_order(self):
        """Evaluate if player's prepared food matches desired food"""
        is_correct = (self.player_prepared_food == self.desired_food)
        self.current_phase = "completed" if is_correct else "failed"
        return is_correct

class CookingGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Cooking Game - Restaurant Management")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 32)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Game queues
        self.waiting_to_order = []
        self.waiting_to_pickup = []
        self.completed_orders = 0
        self.failed_orders = 0
        self.next_customer_id = 1
        
        # Game state
        self.game_over = False
        self.score = 0
        
        # Customer spawning
        self.last_customer_spawn = time.time()
        self.customer_spawn_interval = 10  # seconds
        
    def draw_button(self, x, y, width, height, color, text, text_color=BLACK):
        """Draw a button with text"""
        pygame.draw.rect(self.screen, color, (x, y, width, height), border_radius=8)
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height), 2, border_radius=8)
        
        text_surf = self.font.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=(x + width/2, y + height/2))
        self.screen.blit(text_surf, text_rect)
        
        return pygame.Rect(x, y, width, height)
    
    def draw_progress_bar(self, x, y, width, height, progress, color):
        """Draw a progress bar"""
        # Background
        pygame.draw.rect(self.screen, GRAY, (x, y, width, height), border_radius=4)
        # Progress
        if progress > 0:
            pygame.draw.rect(self.screen, color, (x, y, width * progress, height), border_radius=4)
        # Border
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height), 1, border_radius=4)
    
    def draw_customer(self, customer, x, y, width=250):
        """Draw a customer card"""
        # Card background based on phase
        if customer.current_phase == "waiting_to_order":
            card_color = LIGHT_GREEN
            status_text = "Waiting to Order"
            timer_color = BLUE
        elif customer.current_phase == "waiting_to_pickup":
            card_color = YELLOW
            status_text = "Waiting for Food"
            timer_color = ORANGE
        elif customer.current_phase == "completed":
            card_color = GREEN
            status_text = "Order Complete!"
            timer_color = GREEN
        else:  # failed
            card_color = LIGHT_RED
            status_text = "Order Failed"
            timer_color = RED
        
        # Draw customer card
        pygame.draw.rect(self.screen, card_color, (x, y, width, 80), border_radius=10)
        pygame.draw.rect(self.screen, BLACK, (x, y, width, 80), 2, border_radius=10)
        
        # Customer info
        customer_text = self.font.render(f"Customer #{customer.id}", True, BLACK)
        self.screen.blit(customer_text, (x + 10, y + 10))
        
        status_text = self.small_font.render(status_text, True, BLACK)
        self.screen.blit(status_text, (x + 10, y + 35))
        
        # Food info
        if customer.current_phase == "waiting_to_order":
            food_text = self.small_font.render("???", True, BLACK)
        else:
            food_text = self.small_font.render(f"Wants: {customer.desired_food}", True, BLACK)
        self.screen.blit(food_text, (x + 10, y + 55))
        
        # Timer
        if customer.current_phase in ["waiting_to_order", "waiting_to_pickup"]:
            time_text = self.font.render(f"{int(customer.time_remaining)}s", True, timer_color)
            self.screen.blit(time_text, (x + width - 40, y + 10))
            
            # Progress bar
            max_time = ORDER_TIME if customer.current_phase == "waiting_to_order" else PREP_TIME
            progress = customer.time_remaining / max_time
            self.draw_progress_bar(x + 10, y + 75, width - 20, 5, progress, timer_color)
    
    def spawn_customer(self):
        """Spawn a new customer in waiting to order queue"""
        current_time = time.time()
        if current_time - self.last_customer_spawn >= self.customer_spawn_interval:
            self.last_customer_spawn = current_time
            
            customer = Customer(self.next_customer_id)
            self.waiting_to_order.append(customer)
            self.next_customer_id += 1
            return True
        return False
    
    def take_customer_order(self, customer_index):
        """Take order from customer (move to waiting_to_pickup)"""
        if 0 <= customer_index < len(self.waiting_to_order):
            customer = self.waiting_to_order[customer_index]
            if customer.take_order():
                self.waiting_to_order.pop(customer_index)
                self.waiting_to_pickup.append(customer)
                return True
        return False
    
    def fulfill_order(self, customer_index, food_index):
        """Fulfill order for customer"""
        if 0 <= customer_index < len(self.waiting_to_pickup):
            customer = self.waiting_to_pickup[customer_index]
            prepared_food = FOOD_ITEMS[food_index]
            if customer.fulfill_order(prepared_food):
                self.waiting_to_pickup.pop(customer_index)
                if customer.current_phase == "completed":
                    self.completed_orders += 1
                    self.score += 10
                else:
                    self.failed_orders += 1
                return True
        return False
    
    def update_game(self):
        """Update game state"""
        if self.game_over:
            return
        
        # Spawn new customers
        self.spawn_customer()
        
        # Update customers and check for timeouts
        # Check waiting_to_order queue
        expired_customers = []
        for i, customer in enumerate(self.waiting_to_order):
            if customer.update_timer() and customer.current_phase == "waiting_to_order":
                expired_customers.append(i)
        
        # Remove expired customers from waiting_to_order (reverse to maintain indices)
        for i in sorted(expired_customers, reverse=True):
            customer = self.waiting_to_order.pop(i)
            customer.current_phase = "failed"
            self.failed_orders += 1
        
        # Check waiting_to_pickup queue
        expired_customers = []
        for i, customer in enumerate(self.waiting_to_pickup):
            if customer.update_timer() and customer.current_phase == "waiting_to_pickup":
                expired_customers.append(i)
        
        # Remove expired customers from waiting_to_pickup
        for i in sorted(expired_customers, reverse=True):
            customer = self.waiting_to_pickup.pop(i)
            customer.current_phase = "failed"
            self.failed_orders += 1
        
        # Check game over condition
        if self.failed_orders >= MAX_FAILED_ORDERS:
            self.game_over = True
    
    def draw(self):
        """Draw the game"""
        self.screen.fill(WHITE)
        
        # Draw header
        title = self.title_font.render("Restaurant Cooking Game", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 20))
        
        # Draw score and stats
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (50, 80))
        
        completed_text = self.font.render(f"Completed: {self.completed_orders}", True, GREEN)
        self.screen.blit(completed_text, (200, 80))
        
        failed_text = self.font.render(f"Failed: {self.failed_orders}/{MAX_FAILED_ORDERS}", True, RED)
        self.screen.blit(failed_text, (400, 80))
        
        # Draw queues
        order_title = self.font.render("WAITING TO ORDER (Click to Take Order)", True, BLUE)
        self.screen.blit(order_title, (50, 120))
        
        pickup_title = self.font.render("WAITING FOR PICKUP (Select Food to Serve)", True, ORANGE)
        self.screen.blit(pickup_title, (500, 120))
        
        # Draw waiting to order queue
        for i, customer in enumerate(self.waiting_to_order):
            self.draw_customer(customer, 50, 150 + i * 100)
        
        # Draw waiting to pickup queue
        for i, customer in enumerate(self.waiting_to_pickup):
            self.draw_customer(customer, 500, 150 + i * 100)
        
        # Draw food selection buttons
        food_title = self.font.render("PREPARE FOOD:", True, PURPLE)
        self.screen.blit(food_title, (50, 550))
        
        # Draw food buttons in two rows
        for i, food in enumerate(FOOD_ITEMS[:8]):  # First 8 foods
            x = 50 + (i % 4) * 180
            y = 580 + (i // 4) * 40
            self.draw_button(x, y, 170, 30, LIGHT_GREEN, food)
        
        # Draw game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.title_font.render("GAME OVER", True, RED)
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
            
            final_score = self.font.render(f"Final Score: {self.score} | Completed Orders: {self.completed_orders}", True, WHITE)
            self.screen.blit(final_score, (SCREEN_WIDTH//2 - final_score.get_width()//2, SCREEN_HEIGHT//2))
            
            restart_text = self.font.render("Press R to Restart or Q to Quit", True, WHITE)
            self.screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle game events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_r and self.game_over:
                    self.__init__()  # Restart game
            
            if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check waiting to order queue clicks
                for i, customer in enumerate(self.waiting_to_order):
                    customer_rect = pygame.Rect(50, 150 + i * 100, 250, 80)
                    if customer_rect.collidepoint(mouse_pos):
                        self.take_customer_order(i)
                        break
                
                # Check food button clicks for waiting to pickup queue
                if self.waiting_to_pickup:
                    for i, food_index in enumerate(range(min(8, len(FOOD_ITEMS)))):
                        x = 50 + (i % 4) * 180
                        y = 580 + (i // 4) * 40
                        food_rect = pygame.Rect(x, y, 170, 30)
                        
                        if food_rect.collidepoint(mouse_pos):
                            # Fulfill the first order in waiting_to_pickup with selected food
                            self.fulfill_order(0, food_index)
                            break
        
        return True
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_events()
            self.update_game()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = CookingGame()
    game.run()