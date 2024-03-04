import random
from collections import deque
from abc import ABC, abstractmethod

class GameState:
    def __init__(self, n):
        self.n = n
        self.available_numbers = set(range(1, n + 1))
        self.player1_score = 0
        self.player2_score = 0

    def copy(self):
        # Corrected copy method, directly assigning values to attributes
        copy_state = GameState(self.n)  # Create an empty GameState object
        copy_state.available_numbers = self.available_numbers.copy()  # Assign copied available numbers
        copy_state.player1_score = self.player1_score
        copy_state.player2_score = self.player2_score
        return copy_state

class Strategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_move(self, game_state: GameState):
        pass

    def get_candidate_numbers(self, game_state: GameState):
        pass

class BackwardInductionStrategy(Strategy):
    def __init__(self, maximizing_player=True):
        super().__init__()
        self.maximizing_player = maximizing_player

    def make_move(self, game_state: GameState):
        if not game_state.available_numbers:
            print("Error: No available numbers")
            return None

        # Create a queue to store game states
        queue = deque([game_state.copy()])

        # Alpha-beta pruning variables
        alpha = float("-inf")  # Best score for maximizing player
        beta = float("inf")  # Best score for minimizing player

        while queue:
            current_state = queue.popleft()

            # Check all possible moves and their outcomes
            for number in current_state.available_numbers:
                new_state = current_state.copy()
                new_state.available_numbers.remove(number)
                new_state_score = new_state.player1_score + number

                # Check for base case (opponent has no moves)
                opponent_numbers = new_state.available_numbers.copy()  # Use a copy to avoid modifying original
                if not opponent_numbers:
                    score = self.player1_score  # Player's score (already maximized)
                    new_state.player1_score = score
                    queue.append(new_state)
                    continue

                # Check if the game is over for the opponent
                if not new_state.available_numbers:
                    # If maximizing player, score is the current player's score
                    # If minimizing player, score is the opponent's score (0)
                    score = new_state_score if self.maximizing_player else 0
                    new_state.player1_score = score
                    queue.append(new_state)
                    continue

                # Simulate the opponent's move (using updated _simulate_opponent_move)
                opponent_score = self._simulate_opponent_move(opponent_numbers)
                new_state.player2_score = opponent_score

                # Add the simulated state to the queue (after evaluating the opponent's move)
                if self.maximizing_player:
                    score = max(new_state_score - opponent_score, current_state.player1_score)
                    # Apply alpha-beta pruning
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break  # Prune branches that won't be explored
                else:
                    score = min(new_state_score - opponent_score, current_state.player1_score)
                    # Apply alpha-beta pruning
                    beta = min(beta, score)
                    if beta <= alpha:
                        break  # Prune branches that won't be explored

                new_state.player1_score = score
                queue.append(new_state)

        # Choose the move with the highest score (or lowest for minimizing player)
        try:
            best_state = max(queue, key=lambda x: x.player1_score)
            # Choose the return value based on your application's logic:
            # 1. Return the chosen number: return best_state.player1_score - game_state.player1_score
            # 2. Return the score gained by the move: return best_state.available_numbers
        except ValueError:
            print("Error: No valid moves found")
            return None
     
    def _simulate_opponent_move(self, available_numbers):
        opponent_score, _ = min(
            ((number, available_numbers.copy() - {number})  # Create remaining_numbers within the loop
            for number in available_numbers),
            key=lambda x: x[0],  # Choose the move that minimizes opponent's score
        )
        return opponent_score

class ChooseExtremeNumberStrategy(Strategy):
    def __init__(self, choose_max=True):
        super().__init__()
        self.choose_max = choose_max

    def get_candidate_numbers(self, game_state: GameState):
        available_numbers = game_state.available_numbers
        if available_numbers:
            if self.choose_max:
                return [max(available_numbers)]
            else:
                return [min(available_numbers)]
        else:
            return []

    def make_move(self, game_state: GameState):
        available_numbers = game_state.available_numbers
        player_score = game_state.player1_score
        opponent_score = game_state.player2_score

        if available_numbers:
            if self.choose_max:
                return max(available_numbers)
            else:
                return min(available_numbers)
        else:
            return None

class DynamicStrategy(Strategy):
    def get_candidate_numbers(self, game_state: GameState):
        available_numbers = game_state.available_numbers
        player_score = game_state.player1_score  # Adjust for player 2 if needed
        opponent_score = game_state.player2_score  # Adjust for player 1 if needed

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

    def make_move(self, game_state: GameState):
        candidates = self.get_candidate_numbers(game_state)
        return random.choice(candidates) if candidates else None

class MaximizeLeadStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, game_state: GameState):
        available_numbers = game_state.available_numbers
        player_score = game_state.player1_score  # Adjust for player 2 if needed
        opponent_score = game_state.player2_score  # Adjust for player 1 if needed

        # Calculate lead
        lead = player_score - opponent_score

        # Initialize variables for tracking
        best_move = None
        max_lead_after_move = float("-inf")  # Negative infinity

        # Find the move that maximizes the lead
        for number in available_numbers:
            potential_lead = lead + number  # Calculate potential lead
            if potential_lead > max_lead_after_move:
                max_lead_after_move = potential_lead
                best_move = number

        return best_move

class MinimizeLeadStrategy(Strategy):
    def make_move(self, game_state: GameState):
        player_score = game_state.player1_score
        opponent_score = game_state.player2_score

        # Introduce a counter to limit loop iterations
        max_iterations = len(game_state.available_numbers)  # Limit based on available numbers
        iteration = 0

        while iteration < max_iterations:
            number = random.choice(list(game_state.available_numbers))
            if player_score + number <= opponent_score:
                return number

            iteration += 1

        # If no suitable number is found, choose a number that minimizes lead increase
        return min(game_state.available_numbers)

class RandomizedStrategy(Strategy):
    def get_candidate_numbers(self, game_state: GameState):
        player_score = game_state.player1_score
        opponent_score = game_state.player2_score
        available_numbers = game_state.available_numbers
        available_subsets = [subset for subset in available_numbers if subset >= opponent_score - player_score]
        return available_subsets

    def make_move(self, game_state: GameState):
        candidates = self.get_candidate_numbers(game_state)
        return random.choice(candidates) if candidates else None

class StrategyFactory:
    def __init__(self):
        self.strategies = {
            "1": ChooseExtremeNumberStrategy(choose_max=True),
            "2": ChooseExtremeNumberStrategy(choose_max=False),
            "3": DynamicStrategy(),
            "4": RandomizedStrategy(),
            "5": BackwardInductionStrategy(),
            "6": MinimizeLeadStrategy(),
            "7": MaximizeLeadStrategy(),
        }

    def get_strategy(self, choice):
        if choice in self.strategies:
            return self.strategies[choice]
        else:
            raise ValueError(f"Invalid strategy choice: {choice}")

class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.strategy = None

    def choose_strategy(self):
        factory = StrategyFactory()

        while True:
            print("Available strategies:")
            for key, strategy in factory.strategies.items():
                print(f"{key}: {strategy.__class__.__name__}")

            choice = input("Choose your strategy (1-7): ")
            try:
                self.strategy = factory.get_strategy(choice)
                break
            except ValueError as e:
                print(e)

    def make_move(self, game_state: GameState):
        if self.strategy is None:
            self.choose_strategy()
        try:
            return self.strategy.make_move(game_state.copy())  # Pass a copy of the game state
        except (TypeError, ValueError, AttributeError) as e:
            print(f"Error making move: {e}")
            return None

class CatchUpGame:
    def __init__(self, n, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.game_state = GameState(n)

    def is_game_over(self):
        return not self.game_state.available_numbers

    def get_winner(self):
        if self.player1.score > self.player2.score:
            return self.player1
        elif self.player2.score > self.player1.score:
            return self.player2
        else:
            return None

    def play_game(self):
        while not self.is_game_over():
            print(f"Available numbers: {sorted(list(self.game_state.available_numbers))}")

            # Player 1's turn
            p1_choice = self.player1.make_move(self.game_state.copy())  # Pass a copy of game state
            if p1_choice is not None:
                self.game_state.available_numbers.remove(p1_choice)
                self.player1.score += p1_choice
                print(f"{self.player1.name} chooses: {p1_choice}")
            else:
                break

            # Player 2's turn
            p2_choice = self.player2.make_move(self.game_state.copy())  # Pass a copy of game state
            if p2_choice is not None:
                self.game_state.available_numbers.remove(p2_choice)
                self.player2.score += p2_choice
                print(f"{self.player2.name} chooses: {p2_choice}")
            else:
                break

            print(f"{self.player1.name} score: {self.player1.score}, {self.player2.name} score: {self.player2.score}\n")

        winner = self.get_winner()
        if winner:
            print(f"{winner.name} wins!")
        else:
            print("It's a tie!")

# Try with n=5 and two players with different strategies
if __name__ == "__main__":
    player1 = Player("Player 1")
    player2 = Player("Player 2")

    catch_up_game = CatchUpGame(25, player1, player2)
    catch_up_game.play_game()
