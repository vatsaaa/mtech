import random

class Strategy:
    def __init__(self):
        pass

    def make_move(self, available_numbers, player_score, opponent_score):
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

    def make_move(self, available_numbers, player_score, opponent_score):
        if available_numbers:
            if self.choose_max:
                return max(available_numbers)
            else:
                return min(available_numbers)
        else:
            return None

class DynamicStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Choose a number based on the current state
        if player_score == opponent_score:
            # If scores are equal, choose randomly
            return random.choice(available_numbers)
        elif player_score < opponent_score:
            # If behind, choose a number that minimizes the lead
            return min(available_numbers, key=lambda x: abs((player_score + x) - opponent_score))
        else:
            # If ahead, choose a number that maximizes the lead
            return max(available_numbers, key=lambda x: abs((player_score + x) - opponent_score))

class MaximizeNumbersStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Choose the maximum available number
        if available_numbers:
            return max(available_numbers)
        else:
            return None

class MinimizeNumbersStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Choose the minimum available number
        if available_numbers:
            return min(available_numbers)
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

class MaximizeLeadStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        # Choose the number that maximizes the lead over opponent's last move
        if available_numbers:
            return max(available_numbers, key=lambda x: abs((player_score + x) - opponent_score))
        else:
            return None

class RandomizedStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def make_move(self, available_numbers, player_score, opponent_score):
        available_subsets = [subset for subset in available_numbers if subset >= opponent_score - player_score]
        if available_subsets:
            return random.choice(available_subsets)
        else:
            return None

class Player:
    def __init__(self, name, strategy):
        self.name = name
        self.score = 0
        self.strategy = strategy

    def make_move(self, available_numbers, opponent_score):
        return self.strategy.make_move(available_numbers, self.score, opponent_score)

class CatchUpGame:
    def __init__(self, n, player1, player2):
        self.n = n
        self.available_numbers = set(range(1, n + 1))
        self.player1 = player1
        self.player2 = player2

    def play_game(self):
        while self.available_numbers:
            print(f"Available numbers: {sorted(list(self.available_numbers))}")

            # Player 1's turn
            p1_choice = self.player1.make_move(self.available_numbers, self.player2.score)
            if p1_choice is not None:
                self.available_numbers -= {p1_choice}
                self.player1.score += p1_choice
                print(f"{self.player1.name} chooses: {p1_choice}")
            else:
                break

            # Player 2's turn
            p2_choice = self.player2.make_move(self.available_numbers, self.player1.score)
            if p2_choice is not None:
                self.available_numbers -= {p2_choice}
                self.player2.score += p2_choice
                print(f"{self.player2.name} chooses: {p2_choice}")
            else:
                break

            print(f"{self.player1.name} score: {self.player1.score}, {self.player2.name} score: {self.player2.score}\n")

        if self.player1.score > self.player2.score:
            print(f"{self.player1.name} wins!")
        elif self.player2.score > self.player1.score:
            print(f"{self.player2.name} wins!")
        else:
            print("It's a tie!")

# Try with n=5 and two players with different strategies
if __name__ == "__main__":
    player1 = Player("Player 1", RandomizedStrategy())
    player2 = Player("Player 2", RandomizedStrategy())

    catch_up_game = CatchUpGame(25, player1, player2)
    catch_up_game.play_game()

