import pygame
import sys

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Text-Based Cooking Challenge")

clock = pygame.time.Clock()

COLOR_BACKGROUND = (250, 240, 230)  
COLOR_TEXT = (80, 45, 30)           
COLOR_TIMER = (50, 120, 50)         
COLOR_TIMER_LOW = (200, 40, 40)     
COLOR_CORRECT = (0, 150, 0)         
COLOR_INCORRECT = (200, 40, 40)     

font_question = pygame.font.Font(None, 48)
font_input = pygame.font.Font(None, 42)
font_timer = pygame.font.Font(None, 36)
font_feedback = pygame.font.Font(None, 52)


question_text = "What is the capital of France?"
correct_answer = "paris"
time_limit_seconds = 5 

menu = ["apple pie", "pumpkin pie", "espresso", "chai latte", "pumpkin spice latte", "cinnamon roll", "blueberry muffin", "banana bread"]

# ingredients: milk, sugar, flour, eggs, cinnamon, pumpkin, apple, banana, blueberry, tea leaves, coffee
# grover's search to find ingredients for a menu item
# each searching action takes a click that makes the items appear
# dragging and dropping ingredients to a bowl to prepare the dish, each drop of ingredient applies a set of gates for another algorithm step to "cook"

user_input = ""
game_state = "playing"  
start_ticks = pygame.time.get_ticks() 
feedback_message = ""


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN and game_state == "playing":
            if event.key == pygame.K_RETURN: 
                if user_input.lower() == correct_answer:
                    game_state = "correct"
                    feedback_message = "Correct! 🎉"
                else:
                    game_state = "incorrect"
                    feedback_message = "Sorry, that's not the right answer."
            elif event.key == pygame.K_BACKSPACE:
                user_input = user_input[:-1] 
            else:
                user_input += event.unicode
    
    if game_state == "playing":
        
        seconds_passed = (pygame.time.get_ticks() - start_ticks) / 1000
        time_left = time_limit_seconds - seconds_passed
        
        if time_left <= 0:
            game_state = "times_up"
            feedback_message = "Sorry, you ran out of time!"
            time_left = 0 
    
    screen.fill(COLOR_BACKGROUND)

    
    question_surface = font_question.render(question_text, True, COLOR_TEXT)
    screen.blit(question_surface, (SCREEN_WIDTH // 2 - question_surface.get_width() // 2, 150))
    
    
    timer_color = COLOR_TIMER if time_left > 2 else COLOR_TIMER_LOW
    timer_surface = font_timer.render(f"Time: {int(time_left)}", True, timer_color)
    screen.blit(timer_surface, (20, 20))
    
    
    input_prompt_surface = font_input.render("> ", True, COLOR_TEXT)
    input_surface = font_input.render(user_input, True, COLOR_TEXT)
    
    
    total_input_width = input_prompt_surface.get_width() + input_surface.get_width()
    input_start_x = SCREEN_WIDTH // 2 - total_input_width // 2
    
    screen.blit(input_prompt_surface, (input_start_x, 250))
    screen.blit(input_surface, (input_start_x + input_prompt_surface.get_width(), 250))

    
    if game_state != "playing":
        if game_state == "correct":
            feedback_color = COLOR_CORRECT
        else: 
            feedback_color = COLOR_INCORRECT
            
        feedback_surface = font_feedback.render(feedback_message, True, feedback_color)
        screen.blit(feedback_surface, (SCREEN_WIDTH // 2 - feedback_surface.get_width() // 2, 350))


    
    pygame.display.flip()
    
    
    clock.tick(60)


pygame.quit()
sys.exit()