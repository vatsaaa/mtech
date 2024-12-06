from typing import List, Tuple
import random
import itertools

def calculate_score(numbers: List[int]) -> int:
    return sum(numbers)

def get_player1_strategy() -> Tuple[int, bool]:
    while True:
        strategy = input("Select strategy for Player 1 (0 for maximizer, 1 for minimizer): ")
        alpha_beta = input("Use alpha-beta pruning? (Y/N): ").upper()
        if strategy in ['0', '1'] and alpha_beta in ['Y', 'N']:
            return int(strategy), alpha_beta == 'Y'
        else:
            print("Invalid input. Please enter 0 for maximizer or 1 for minimizer, and Y/N for alpha-beta pruning.")

def random_choice(numbers: List[int], current_score: int) -> int:
    if len(numbers) == 1:  # Handle the case where only one number is left
        return numbers[0]
    subsets = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers) + 1):
            subset = numbers[i:j]
            if sum(subset) >= current_score:
                subsets.append(subset)
    if not subsets:  # If no subset found, choose the smallest number
        return min(numbers)
    chosen_subset = random.choice(subsets)
    return random.choice(chosen_subset)  # Choose a random number from the chosen subset

def print_possible_moves(numbers: List[int], current_score: int, player: str) -> None:
    print(f"Player {player}: Possible moves: {numbers}")

def minimax_choice(numbers: List[int], current_score: int, is_maximizer: bool, alpha: int, beta: int, player: str, use_alpha_beta: bool) -> int:
    if len(numbers) == 1:  # Handle the case where only one number is left
        return numbers[0]
    if is_maximizer:
        best_score = float('-inf')
        best_move = None
        for num in numbers:
            remaining_numbers = [x for x in numbers if x != num]
            score = num + calculate_score(remaining_numbers)
            if use_alpha_beta:
                print(f"Alpha-Beta Pruning: Player {player}, Move: {num}, Alpha: {alpha}, Beta: {beta}, Score: {score}")
            if score > best_score:
                best_score = score
                best_move = num
                if use_alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
        return best_move
    else:
        worst_score = float('inf')
        worst_move = None
        for num in numbers:
            remaining_numbers = [x for x in numbers if x != num]
            score = num + calculate_score(remaining_numbers)
            if use_alpha_beta:
                print(f"Alpha-Beta Pruning: Player {player}, Move: {num}, Alpha: {alpha}, Beta: {beta}, Score: {score}")
            if score < worst_score:
                worst_score = score
                worst_move = num
                if use_alpha_beta:
                    beta = min(beta, worst_score)
                    if beta <= alpha:
                        break
        return worst_move

def get_possible_moves(numbers: List[int], prev_player_choice_sum: int) -> List[int]:
    choices = []
    for i in range(1, len(numbers) + 1):
        for subset in itertools.combinations(numbers, i):
            if sum(subset) >= prev_player_choice_sum:
                choices.append(subset)
    return choices

def play_game() -> None:
    n = int(input("Enter the highest number for the game: "))
    if n < 1:
        print("The highest number must be at least 1.")
        return
    
    strategy_player1, use_alpha_beta = get_player1_strategy()
    strategy_player2 = 1 - strategy_player1  # Opposite strategy for Player 2
    
    numbers = list(range(1, n + 1))  # Initial numbers
    print(f"Initial numbers: {numbers}")
    
    player1_scores: List[int] = []
    player2_scores: List[int] = []
    
    # Player 1's first choice
    player1_choice = random.choice(numbers)
    player1_scores.append(player1_choice)
    print(f"Player 1 chooses: {player1_choice} (Current score: {calculate_score(player1_scores)})")
    numbers.remove(player1_choice)
    
    alpha = float('-inf')
    beta = float('inf')
    
    while numbers:
        # Player 2's first choice
        possible_moves = get_possible_moves(numbers, calculate_score(player1_scores))
        print_possible_moves(possible_moves, player1_choice, "2")
        player2_choice = minimax_choice(possible_moves, player1_choice, bool(strategy_player2), alpha, beta, "2", use_alpha_beta)
        player2_scores.append(player2_choice)
        print(f"Player 2 chooses: {player2_choice} (Current score: {calculate_score(player2_scores)})")
        if player2_choice in numbers:
            numbers.remove(player2_choice)
        
        if not numbers:
            break
        
        # Player 1's second choice
        print_possible_moves(numbers, sum(player2_scores), "1")
        player1_choice = minimax_choice(numbers, sum(player2_scores), bool(strategy_player1), alpha, beta, "1", use_alpha_beta)
        player1_scores.append(player1_choice)
        print(f"Player 1 chooses: {player1_choice} (Current score: {calculate_score(player1_scores)})")
        if player1_choice in numbers:
            numbers.remove(player1_choice)
        
        if not numbers:
            break
        
        # Player 2's second choice
        print_possible_moves(numbers, sum(player1_scores), "2")
        player2_choice = minimax_choice(numbers, sum(player1_scores), bool(strategy_player2), alpha, beta, "2", use_alpha_beta)
        player2_scores.append(player2_choice)
        print(f"Player 2 chooses: {player2_choice} (Current score: {calculate_score(player2_scores)})")
        if player2_choice in numbers:
            numbers.remove(player2_choice)
        
    player1_total_score = calculate_score(player1_scores)
    player2_total_score = calculate_score(player2_scores)
    
    if player1_total_score > player2_total_score:
        print(f"\nResult -> Player 1: {player1_total_score} and Player 2: {player2_total_score}. Player 1 wins!")
    elif player1_total_score < player2_total_score:
        print(f"\nResult -> Player 1: {player1_total_score} and Player 2: {player2_total_score}. Player 2 wins!")
    else:
        print("\nResult -> It's a draw!")

# Play the game
play_game()
