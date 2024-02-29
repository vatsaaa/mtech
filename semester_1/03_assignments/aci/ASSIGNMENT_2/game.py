import random

class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0

    def make_move(self, available_numbers, opponent_score):
        available_subsets = [subset for subset in available_numbers if subset >= opponent_score - self.score]
        if available_subsets:
            chosen_subset = random.choice(available_subsets)
            return chosen_subset
        else:
            return None

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

            print(f"{self.player1.name} score: {self.player1.score}")
            print(f"{self.player2.name} score: {self.player2.score}\n")

        if self.player1.score > self.player2.score:
            print(f"{self.player1.name} wins!")
        elif self.player2.score > self.player1.score:
            print(f"{self.player2.name} wins!")
        else:
            print("It's a tie!")

# Example with n=5 and two players
player1 = Player("Player 1")
player2 = Player("Player 2")

catch_up_game = CatchUpGame(5, player1, player2)
catch_up_game.play_game()
