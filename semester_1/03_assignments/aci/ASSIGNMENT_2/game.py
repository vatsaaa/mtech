import random
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
    def make_move(self, available_numbers, player_score, opponent_score):
        pass

    def get_candidate_numbers(self, available_numbers, player_score, opponent_score):
        # Generate potential choices based on the strategy
        pass

class BackwardInductionStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Implement a simplified backward induction strategy
        if sum(available_numbers) % 2 == 0:
            # If the sum is even, aim for a draw
            return min(available_numbers)
        else:
            # If the sum is odd, aim for a win
            return max(available_numbers)

class ChooseExtremeNumberStrategy(Strategy):
    def __init__(self, choose_max=True):
        super().__init__()
        self.choose_max = choose_max

    def get_candidate_numbers(self, available_numbers, player_score, opponent_score):
        if available_numbers:
            if self.choose_max:
                return [max(available_numbers)]
            else:
                return [min(available_numbers)]
        else:
            return []

    def make_move(self, available_numbers, player_score, opponent_score):
        candidates = self.get_candidate_numbers(available_numbers.copy(), player_score, opponent_score)
        return candidates[0] if candidates else None

class DynamicStrategy(Strategy):
    def get_candidate_numbers(self, available_numbers, player_score, opponent_score):
        if player_score == opponent_score:
            # If scores are equal, choose all available numbers
            return list(available_numbers)
        elif player_score < opponent_score:
            # If behind, choose numbers that minimize the lead
            return list(available_numbers)
        else:
            # If ahead, choose any number (potentially can be further refined)
            return list(available_numbers)

    def make_move(self, available_numbers, player_score, opponent_score):
        candidates = self.get_candidate_numbers(available_numbers.copy(), player_score, opponent_score)
        return random.choice(candidates) if candidates else None

class MaximizeLeadStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Choose the number that maximizes the lead over opponent's last move
        if available_numbers:
            return max(available_numbers, key=lambda x: abs((player_score + x) - opponent_score))
        else:
            return None

class MinimizeLeadStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Choose the number that minimizes the lead over opponent's last move
        if available_numbers:
            return min(available_numbers, key=lambda x: abs((player_score + x) - opponent_score))
        else:
            return None

class RandomizedStrategy(Strategy):
    def get_candidate_numbers(self, available_numbers, player_score, opponent_score):
        available_subsets = [subset for subset in available_numbers if subset >= opponent_score - player_score]
        return available_subsets

    def make_move(self, available_numbers, player_score, opponent_score):
        candidates = self.get_candidate_numbers(available_numbers.copy(), player_score, opponent_score)
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

    def make_move(self, available_numbers, opponent_score):
        if self.strategy is None:
            self.choose_strategy()
        try:
            return self.strategy.make_move(available_numbers.copy(), self.score, opponent_score)
        except (TypeError, ValueError) as e:
            print(f"Error making move: {e}")
            return None

class CatchUpGame:
    def __init__(self, n, player1, player2):
        self.n = n
        self.available_numbers = set(range(1, n + 1))
        self.player1 = player1
        self.player2 = player2

    def is_game_over(self):
        return not self.available_numbers

    def get_winner(self):
        if self.player1.score > self.player2.score:
            return self.player1
        elif self.player2.score > self.player1.score:
            return self.player2
        else:
            return None

    def play_game(self):
        while not self.is_game_over():
            print(f"Available numbers: {sorted(list(self.available_numbers))}")

            # Player 1's turn
            p1_choice = self.player1.make_move(
                self.available_numbers.copy(), self.player2.score
            )
            if p1_choice is not None:
                self.available_numbers.remove(p1_choice)
                self.player1.score += p1_choice
                print(f"{self.player1.name} chooses: {p1_choice}")
            else:
                break

            # Player 2's turn
            p2_choice = self.player2.make_move(
                self.available_numbers.copy(), self.player1.score
            )
            if p2_choice is not None:
                self.available_numbers.remove(p2_choice)
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
