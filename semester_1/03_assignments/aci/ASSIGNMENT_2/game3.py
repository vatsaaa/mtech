import random
from collections import deque

class GameState:
    def __init__(self, n):
        self.n = n
        self.available_numbers = set(range(1, n + 1))
        self.current_player_score = 0
        self.other_player_score = 0

    def copy(self):
        copy_state = GameState(self.n)  # Create an empty GameState object
        copy_state.available_numbers = self.available_numbers.copy()  # Assign copied available numbers
        copy_state.current_player_score = self.current_player_score
        copy_state.other_player_score = self.other_player_score
        return copy_state

"""
MinimaxStrategy class has the combined logic of both Minimax and ⍺-β pruning techniques,
it also incorporates the backward induction approach to find the best move for the player
"""
class MinimaxStrategy():
    def __init__(self, maximizing_player=True, max_depth=5):
        self.maximizing_player = maximizing_player
        self.max_depth = max_depth

    def set_game_state(self, game_state: GameState):
        self.game_state = game_state.copy()  

    def get_candidate_numbers(self, game_state):
        available_numbers = game_state.available_numbers
        player_score = game_state.current_player_score
        opponent_score = game_state.other_player_score

        if player_score == opponent_score:
            return list(available_numbers)
        elif player_score > opponent_score:
            return sorted(list(available_numbers))
        else:
            return sorted(list(available_numbers), reverse=True) if player_score == 0 else sorted(list(available_numbers))

    def minimax(self, game_state, depth, maximizing=True, alpha=float("-inf"), beta=float("inf")):
        if depth == self.max_depth or not game_state.available_numbers:
            return game_state.current_player_score if maximizing else game_state.other_player_score

        best_score = float("-inf") if maximizing else float("inf")

        for number in game_state.available_numbers:
            new_state = game_state.copy()
            new_state.available_numbers.remove(number)
            new_state.current_player_score += number
            new_state.other_player_score = min(new_state.available_numbers, default=0)

            score = self.minimax(new_state, depth + 1, maximizing=not maximizing, alpha=alpha if maximizing else best_score, beta=beta if not maximizing else best_score)

            if maximizing:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)  # Update alpha only for maximizing player
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)  # Update beta only for minimizing player

        return best_score

class Player:
    def __init__(self, name: str):
        print(name, "ready to play")
        self.name = name
        self.score = 0
        self.strategy = None

    def choose_strategy(self, c: int = None) -> bool:
        try:
            if c is None:
                choice = int(input(f"{self.name}, select (1) to become maximizer OR select (2) to become minimizer? "))
            else:
                choice = c

            if choice not in [1, 2]:
                print(f"Invalid choice. Defaulting {self.name} to maximizer.")
                choice = 1

            self.strategy = MinimaxStrategy(maximizing_player=(choice == 1))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            return False
        else:
            return True

    def make_move(self, game_state: GameState):
        print("Making move for Player ", self.name)
        if self.strategy is None:
            raise ValueError("Player's strategy is not set.")
        
        move = self.strategy.minimax(game_state.copy(), depth=self.strategy.max_depth,
                                     maximizing=self.strategy.maximizing_player, 
                                     alpha=float("-inf"), 
                                     beta=float("inf"))
        return move

class CatchUpGame:
    def __init__(self, n: int, player1: Player, player2: Player):
        print("Game CatchUpWithNumbers initialized...")
        self.game_state = GameState(n)
      
        # Game randomly choose a starting player and sets their preferred strategy
        self.current_player = random.choice([player1, player2])
        print(f"Randomly chose {self.current_player.name} as starting player")
        
        # Each player has a strategy, the random player chooses first
        self.current_player.choose_strategy()
        
        # The other player is automatically assigned the opposite strategy
        self.other_player = player2 if self.current_player == player1 else player1
        c = 2 if self.current_player.strategy == 1 else 1
        self.other_player.choose_strategy(c=c)

    def is_game_over(self):
        return (not self.game_state.available_numbers) 

    def get_winner(self):
        if self.current_player.score > self.other_player.score:
            return self.current_player
        elif self.other_player.score > self.current_player.score:
            return self.other_player
        else:
            return None

    def play(self):
        print("Starting game")
        print(f"Catch-up Numbers: {self.current_player.name} vs. {self.other_player.name}")

        while not self.is_game_over():
            print(f"Available numbers: {sorted(list(self.game_state.available_numbers))}")

            # Current Player's turn
            p1_choice = self.current_player.make_move(self.game_state.copy())
            if p1_choice is not None:
                self.game_state.available_numbers.remove(p1_choice)
                self.current_player.score += p1_choice
                print(f"{self.current_player.name} chooses: {p1_choice}")
            else:
                print(f"{self.current_player.name} cannot make a move.")
                break

            # Switch current player with other player, and print scores
            self.current_player, self.other_player = self.other_player, self.current_player

            print(f"{self.current_player.name} score: {self.current_player.score}, {self.other_player.name} score: {self.other_player.score}\n")

        winner = self.get_winner()
        if winner:
            print(f"{winner.name} wins!")
        else:
            print("It's a tie!")

def main():
    player1 = Player("Player 1")
    player2 = Player("Player 2")

    try:
        n = int(input("Enter the number of numbers (n): "))
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return
    
    if n <= 0:
        print("Invalid input for number of integers to play. Please enter a positive integer.")
        return
    
    catch_up_game = CatchUpGame(n, player1, player2)
    catch_up_game.play()

if __name__ == "__main__":
    main()
