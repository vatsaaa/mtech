import random
from collections import deque
from abc import ABC, abstractmethod

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
GameStrategy (Abstract Base Class): Defines the structure of a game strategy, 
including the abstract methods move() and get_candidate_numbers(). This class
serves as a blueprint for specific strategies.
"""
class GameStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def move(self, game_state: GameState):
        pass

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

"""
MinimaxStrategy class has the combined logic of both Minimax and ⍺-β pruning techniques,
it also incorporates the backward induction approach to find the best move for the player
"""
class MinimaxStrategy(GameStrategy):
    def __init__(self, maximizing_player: bool=True):
        super().__init__()
        self.maximizing_player = maximizing_player

    """
    move() method: applies ⍺-β pruning to efficiently search the game tree and avoid 
    exploring unpromising branches. This method selects the best move for the current
    player based on the Minimax algorithm, considering the opponent's potential moves 
    and maximizing the player's score while minimizing the opponent's score.
    """
    def move(self, game_state: GameState):
        if not game_state.available_numbers:
            print("Error: No available numbers")
            return None

        best_move = None
        alpha = float("-inf")
        beta = float("inf")

        queue = deque([game_state.copy()])

        while queue:
            current_state = queue.popleft()

            for number in current_state.available_numbers:
                new_state = current_state.copy()
                new_state.available_numbers.remove(number)
                new_state_score = new_state.current_player_score + number

                # Check if the game is over for the opponent
                if not new_state.available_numbers:
                    score = new_state_score if self.maximizing_player else 0
                else:
                    # Simulate opponent's move
                    opponent_score = self._simulate_opponent_move(new_state.available_numbers)
                    new_state.other_player_score = opponent_score

                    # Recursively evaluate the next move
                    next_move_score = self._minimax(new_state, alpha, beta, maximizing=not self.maximizing_player)
                    score = new_state_score - opponent_score if self.maximizing_player else next_move_score

                    score, _ = self._alpha_beta_pruning(alpha, beta, score, number)

                # Check if pruning condition is met and break out of the loop if necessary
                if self.maximizing_player:
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
                else:
                    beta = min(beta, score)
                    if beta <= alpha:
                        break

                if score > alpha:
                    alpha = score
                    best_move = number

                queue.append(new_state)

        return best_move

    def _alpha_beta_pruning(self, alpha, beta, score, number):
            if self.maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta:
                    return alpha, None
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    return beta, None
            return score, number

    """
    _minimax() is a private method that performs the recursive evaluation of 
    game states, considering moves of both maximizing and minimizing players
    """
    def _minimax(self, game_state, alpha, beta, maximizing=True):
        if not game_state.available_numbers:
            return game_state.current_player_score if maximizing else 0

        best_score = float("-inf") if maximizing else float("inf")
        for number in game_state.available_numbers:
            new_state = game_state.copy()
            new_state.available_numbers.remove(number)
            new_state_score = new_state.current_player_score + number

            # Simulate opponent's move
            opponent_score = self._simulate_opponent_move(new_state.available_numbers)
            new_state.other_player_score = opponent_score

            score = new_state_score - opponent_score + self._minimax(new_state, alpha, beta, maximizing=not maximizing)
            best_score = max(best_score, score) if maximizing else min(best_score, score)
            alpha = max(alpha, score) if maximizing else min(alpha, score)
            if alpha >= beta:
                break
            if not maximizing and beta <= alpha:
                break
        return best_score

    def _simulate_opponent_move(self, available_numbers):
        if not available_numbers:
            return 0  # Return a default score when no numbers are available
        return min(available_numbers)

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
        
        self.strategy = MinimaxStrategy(True) if choice == 1 else MinimaxStrategy(False)

        return not choice
    
    def get_strategy_name(self) -> str:
        if self.strategy:
            return "Maximizer" if isinstance(self.strategy, MinimaxStrategy) and self.strategy.maximizing_player else "Minimizer"
        else:
            return "No strategy chosen"

    def make_move(self, game_state: GameState):
        try:
            move = self.strategy.move(game_state.copy())
            if move is not None:
                print(f"{self.name} chooses: {move}")
            else:
                print(f"{self.name} cannot make a move.")
            return move
        except (IndexError, ValueError) as e:
            print(f"Error making move: {e}")
            return None

class CatchUpGame:
    def __init__(self, n: int, player1: Player, player2: Player):
        self.game_state = GameState(n)
      
        # Randomly choose the starting player and set their preferred strategy
        self.current_player = random.choice([player1, player2])
        print(f"Randomly chose {self.current_player.name} as starting player")
        
        
        # Choose strategy for the current player
        self.current_player.choose_strategy()
        
        self.other_player = player2 if self.current_player == player1 else player1
        c = 2 if self.current_player.strategy == 1 else 1
        self.other_player.choose_strategy(c=c)

    def is_game_over(self):
        return (not self.game_state.available_numbers) or \
           (self.current_player.strategy.move(self.game_state.copy()) is None and 
            self.other_player.strategy.move(self.game_state.copy()) is None)

    def get_winner(self):
        if self.current_player.score > self.other_player.score:
            return self.current_player
        elif self.other_player.score > self.current_player.score:
            return self.other_player
        else:
            return None
    
    def change_PlayerTurn(self):
        if(self.current_player.name == "Player 1"):
            self.current_player.name = "Player 2"
            self.name = "Player 2"
        else:
            self.current_player.name = "Player 1"
            self.name = "Player 1"

    def play(self):
        print(f"Starting game Catch-up Numbers: {self.current_player.name} vs. {self.other_player.name}")

        while not self.is_game_over():
            print(f"Available numbers: {sorted(list(self.game_state.available_numbers))}")
            self.change_PlayerTurn()
            # Player 1's turn
            p1_choice = self.current_player.make_move(self.game_state.copy())
            if p1_choice is not None:
                self.game_state.available_numbers.remove(p1_choice)
                self.current_player.score += p1_choice
                print(f"{self.current_player.name} chooses: {p1_choice}")
            else:
                print(f"{self.current_player.name} cannot make a move.")
                break

            # Player 2's turn
            p2_choice = self.other_player.make_move(self.game_state.copy())
            if p2_choice is not None:
                self.game_state.available_numbers.remove(p2_choice)
                self.other_player.score += p2_choice
                print(f"{self.other_player.name} chooses: {p2_choice}")
            else:
                print(f"{self.other_player.name} cannot make a move.")
                break

            print(f"{self.current_player.name} score: {self.current_player.score}, {self.other_player.name} score: {self.other_player.score}\n")

        winner = self.get_winner()
        if winner:
            print(f"{winner.name} wins!")
        else:
            print("It's a tie!")

def main():
    player1 = Player("Player 1")
    player2 = Player("Player 2")
    isValidRange = True
    while(isValidRange):
        n = int(input("Enter the range of numbers (n) greater than 1: "))
        if(n>1):
            isValidRange = False
    
    catch_up_game = CatchUpGame(n, player1, player2)
    catch_up_game.play()

if __name__ == "__main__":
    main()


