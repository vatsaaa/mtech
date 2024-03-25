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
    def __init__(self, maximizing_player: bool=True):
        super().__init__()
        self.MAX_DEPTH = 5 
        self.maximizing_player = maximizing_player

    def set_game_state(self, game_state: GameState):
        self.game_state = game_state.copy()  

    def get_candidate_numbers(self, game_state: GameState):
        available_numbers = game_state.available_numbers
        player_score = game_state.current_player_score  # Adjust for player 2 if needed
        opponent_score = game_state.other_player_score  # Adjust for player 1 if needed

        if player_score == opponent_score:
            # If scores are equal, choose all available numbers
            return list(available_numbers)
        elif player_score > opponent_score:
            # When leading, we prioritize choosing smaller numbers:
            #   - Reduce risk of exceeding opponent's score and losing the lead
            #   - Potentially maintain or increase the lead advantage.
            return sorted(list(available_numbers))
        else:
            # If behind, choose all available numbers that don't exceed the opponent's score
            return list(available_numbers)

    def move(self, game_state: GameState):
        if not game_state.available_numbers:
            print("Error: No available numbers")
            return None

        best_move = None
        best_score = float("-inf") if self.maximizing_player else float("inf")

        candidate_numbers = self.get_candidate_numbers(game_state)

        for number in candidate_numbers:
            new_state = game_state.copy()
            new_state.available_numbers.remove(number)
            new_state.current_player_score += number

            # Simulate opponent's move
            if not new_state.available_numbers:
                new_state.other_player_score = 0
            else:
                new_state.other_player_score = min(new_state.available_numbers)

            # Recursively evaluate the next move
            score = self._minimax(new_state, depth=1, maximizing=not self.maximizing_player)

            # Update best move and score
            if (self.maximizing_player and score > best_score) or (not self.maximizing_player and score < best_score):
                best_score = score
                best_move = number

        return best_move

    def _minimax(self, game_state, depth, maximizing=True):
        if depth == self.MAX_DEPTH or not game_state.available_numbers:
            return game_state.current_player_score if maximizing else game_state.other_player_score

        best_score = float("-inf") if maximizing else float("inf")
        for number in game_state.available_numbers:
            new_state = game_state.copy()
            new_state.available_numbers.remove(number)
            new_state.current_player_score += number

            # Simulate opponent's move
            if not new_state.available_numbers:
                new_state.other_player_score = 0
            else:
                new_state.other_player_score = min(new_state.available_numbers)

            # Recursively evaluate the next move
            score = self._minimax(new_state, depth+1, maximizing=not maximizing)

            # Update best score
            if maximizing:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)

        return best_score

class Player:
    def __init__(self, name: str):
        self.name = name
        self.score = 0
        self.strategy = None

    def choose_strategy(self, c: int = None) -> bool:
        if c is None:
            choice = int(input(f"{self.name}, select (1) to become maximizer or (2) to be minimizer? "))
        else:
            choice = c
    
        if choice not in [1, 2]:
            print(f"Invalid choice. Defaulting {self.name} to maximizer.")
            choice = 1
        
        self.strategy = MinimaxStrategy(maximizing_player=(choice == 1))

        return not choice

    def make_move(self, game_state: GameState):
        move = self.strategy.move(game_state.copy())
        return move

class CatchUpGame:
    def __init__(self, n: int, player1: Player, player2: Player):
        self.game_state = GameState(n)
      
        # Randomly choose the starting player and set their preferred strategy
        self.current_player = random.choice([player1, player2])
        print(f"Randomly chose {self.current_player.name} as starting player")
        
        self.current_player.choose_strategy()
        
        self.other_player = player2 if self.current_player == player1 else player1
        c = 2 if self.current_player.strategy == 1 else 1
        self.other_player.choose_strategy(c=c)

    def is_game_over(self):
        return (not self.game_state.available_numbers) or \
            (self.current_player.strategy.move(self.game_state.copy()) is None and \
             self.other_player.strategy.move(self.game_state.copy()) is None)

    def get_winner(self):
        if self.current_player.score > self.other_player.score:
            return self.current_player
        elif self.other_player.score > self.current_player.score:
            return self.other_player
        else:
            return None

    def play(self):
        print(f"Starting game Catch-up Numbers: {self.current_player.name} vs. {self.other_player.name}")

        while not self.is_game_over():
            print(f"Available numbers: {sorted(list(self.game_state.available_numbers))}")

            # Player 1's turn
            p1_choice = self.current_player.make_move(self.game_state.copy())
            if p1_choice is not None:
                self.game_state.available_numbers.remove(p1_choice)
                self.current_player.score += p1_choice
                print(f"{self.current_player.name} chooses: {p1_choice}")
            else:
                print(f"{self.current_player.name} cannot make a move.")
                break

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

    n = int(input("Enter the number of numbers (n): "))
    catch_up_game = CatchUpGame(n, player1, player2)
    catch_up_game.play()

if __name__ == "__main__":
    main()


