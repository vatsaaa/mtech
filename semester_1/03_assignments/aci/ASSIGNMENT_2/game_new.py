import math
import random

def generate_numbers(n):
    """
    Generate a list of numbers from 1 to n.
    """
    return list(range(1, n + 1))

def generate_subsets(numbers, target):
    """
    Generate all subsets of numbers that sum up to at least target.
    """
    subsets = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers) + 1):
            subset = numbers[i:j]
            if sum(subset) >= target:
                subsets.append(subset)
    return subsets

def get_available_moves(numbers, current_sum):
    """
    Get all possible moves given the remaining numbers and current sum.
    """
    available_subsets = generate_subsets(numbers, current_sum)
    available_moves = []
    for subset in available_subsets:
        available_moves.extend(subset)
    available_moves = list(set(available_moves))  # Remove duplicates
    return available_moves

def evaluate_state(current_sum, opponent_sum):
    """
    Evaluate the state based on the current sum and opponent's sum.
    """
    if current_sum > opponent_sum:
        return 1
    elif current_sum == opponent_sum:
        return 0
    else:
        return -1

def minmax_alpha_beta(numbers, current_sum, opponent_sum, is_maximizing_player, alpha, beta):
    """
    Minimax algorithm with alpha-beta pruning for Catch-Up game.
    """
    if not numbers:
        return evaluate_state(current_sum, opponent_sum)

    if is_maximizing_player:
        max_eval = -math.inf
        available_moves = get_available_moves(numbers, opponent_sum)
        for move in available_moves:
            new_numbers = numbers[:]
            new_numbers.remove(move)
            eval = minmax_alpha_beta(new_numbers, current_sum + move, opponent_sum, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        available_moves = get_available_moves(numbers, current_sum)
        for move in available_moves:
            new_numbers = numbers[:]
            new_numbers.remove(move)
            eval = minmax_alpha_beta(new_numbers, current_sum, opponent_sum + move, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(numbers, current_sum, opponent_sum):
    """
    Find the best move for the current player using minimax with alpha-beta pruning.
    """
    best_move = None
    best_eval = -math.inf
    alpha = -math.inf
    beta = math.inf
    available_moves = get_available_moves(numbers, opponent_sum)
    for move in available_moves:
        new_numbers = numbers[:]
        new_numbers.remove(move)
        eval = minmax_alpha_beta(new_numbers, current_sum + move, opponent_sum, False, alpha, beta)
        if eval > best_eval:
            best_eval = eval
            best_move = move
    return best_move

def play_catch_up(n):
    """
    Play the Catch-Up game between two computer players.
    """
    numbers = generate_numbers(n)
    current_sum = 0
    opponent_sum = 0
    while numbers:
        print("Current sum:", current_sum)
        print("Opponent sum:", opponent_sum)
        print("Remaining numbers:", numbers)

        # Player 1 (Computer)
        available_moves_p1 = get_available_moves(numbers, opponent_sum)
        if not available_moves_p1:
            print("P1 has no available moves.")
            break
        p1_move = random.choice(available_moves_p1)
        current_sum += p1_move
        numbers.remove(p1_move)
        print("P1 chooses:", p1_move)
        print("Subsets for P1:", generate_subsets(numbers, current_sum))
        if not numbers:
            break

        # Player 2 (Computer)
        available_moves_p2 = get_available_moves(numbers, current_sum)
        if not available_moves_p2:
            print("P2 has no available moves.")
            break

        p2_move = random.choice(available_moves_p2)
        opponent_sum += p2_move
        numbers.remove(p2_move)
        print("P2 chooses:", p2_move)
        print("Subsets for P2:", generate_subsets(numbers, opponent_sum))
        if not numbers:
            break

    # Evaluate the final result
    if current_sum > opponent_sum:
        print("P1 wins!")
    elif current_sum == opponent_sum:
        print("It's a tie!")
    else:
        print("P2 wins!")

    print("Final scores - P1:", current_sum, "P2:", opponent_sum)

# Example usage
# numbers = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15]
# numbers = generate_numbers(n)
n = int(input("Enter the value of n: "))
# Determine if Player 1 is a maximizer based on user input
player1_maximizer = bool(int(input("Is Player 1 a maximizer? (Enter 1 for Yes, 0 for No): ")))

# Player 2 automatically becomes the minimizer
player2_maximizer = not player1_maximizer
# random.shuffle(numbers)  # Shuffle the numbers for randomness
play_catch_up(n)

