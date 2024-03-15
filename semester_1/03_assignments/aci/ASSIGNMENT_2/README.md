# Introduction
## Problem 1
The provided code implements a two-player game called "Catch-Up with Numbers" using Minimax with alpha-beta pruning for decision making. The code demonstrates the decision-making process involved in the game and highlights the optimization technique used for selecting the best move.


### GameState
Represents the current state of the game, including available numbers, player scores, etc.

### GameStrategy (Abstract Class)
Defines the template for different game strategies. It provides methods for making moves and getting potential numbers to choose from.

### MinimaxStrategy
Implements the Minimax algorithm with alpha-beta pruning. It determines the best move for the current player by considering the opponent's potential moves and maximizing the player's score while minimizing the opponent's score.

### Player
Represents a player in the game, including their name, score, and chosen strategy.

### CatchUpGame
Manages the overall game flow, including determining the starting player, checking the game state, identifying the winner, and handling turns for each player.

### Key Functions:
#### MinimaxStrategy::move( )
This function implements the core Minimax logic with alpha-beta pruning. It explores the game tree using recursion and evaluates different move possibilities for both players.

#### MinimaxStrategy::_alpha_beta_pruning( )
This function helps prune unnecessary branches in the game tree based on alpha and beta values, improving the efficiency of the search.

#### MinimaxStrategy::_minimax( )
This private function recursively evaluates the game state from the perspective of both players, considering their potential moves and scores.

#### Player::choose_strategy( )
This function allows a player to select their strategy (maximizer or minimizer) using user input.

#### Player::make_move( )
This function calls the player's chosen strategy to determine the best move based on the current game state.

#### CatchUpGame::is_game_over( )
This function checks if the game has ended by verifying if there are no more available numbers or neither player can make a valid move.

#### CatchUpGame::get_winner( )
This function determines the winner based on the player's scores.

#### CatchUpGame::play( )
This function handles the entire game loop, including displaying available numbers, prompting players for moves, updating scores, and announcing the winner or a tie.




## Problem 2
WaterSource.pl is Prolog code that defines a set of rules to determine the best water source for a community based on various location factors such as proximity to rivers, lakes, and beaches, rainfall intensity, and aquifer type.
```
         Close to River (< 20 km)
        /                         \
       Yes                         No
      /                           \
   Low Rainfall (Avg. < 150 mm)      Other
  /                              /   \
 Yes (River)                     Medium Rainfall (150-200 mm)
                                  /                         \
                                 Sandy Aquifer (Yes)       No Aquifer
                                /                             \
                             Yes (River)                    No (Rain)
------- High Rainfall (>= 200 mm) --------
                     |
                    Rain
```

### Code walkthrough:
The code starts by defining rules to determine the proximity to water sources. There are several rules: close_to_lake, close_to_river, rainfal_intensity, sandy_aquifier, and close_to_beach. Location rules prompt the user to enter the distance from the location to the nearest lake, river, or beach, respectively. The distance is read from the user using the read predicate. These rules also check if the distance is less than a certain threshold. Rainfall intensity rule checks if the average monthly rainfall amount in millimeters is low, medium, or high. The sandy_aquifer rule prompts the user to answer whether the aquifer is sandy or not. The user's response is read using the read predicate, and the rule checks if the response is "yes". Finally, the code defines rules to predict the best water source based on the decision tree. There are five rules: best_water_source(groundwater), best_water_source(lake), best_water_source(river) (twice), and best_water_source(rain). These rules use the previously defined rules to make predictions. Finally, the main rule is defined. It is the main goal that is called to predict the best water source based on the location factors. It first displays a message to indicate that it is predicting the best water source. Then, it calls the best_water_source rule with a variable WaterSource to store the predicted water source. Finally, it displays the predicted water source to the user.

Next, the code defines rules to determine the rainfall intensity. There are three rules: low_rainfall, medium_rainfall, and high_rainfall. Similar to the previous rules, the user is prompted to enter the average monthly rainfall amount in millimeters. The rainfall amount is read from the user, and these rules check if the amount falls within specific ranges.

The code then defines a rule to determine the aquifer type. The sandy_aquifer rule prompts the user to answer whether the aquifer is sandy or not. The user's response is read using the read predicate, and the rule checks if the response is "yes".

Finally, the code defines rules to predict the best water source based on the decision tree. There are five rules: best_water_source(groundwater), best_water_source(lake), best_water_source(river) (twice), and best_water_source(rain). These rules use the previously defined rules to make predictions.

The best_water_source(groundwater) rule predicts groundwater as the best water source if the location is far from rivers and lakes, and not close to a beach.
The best_water_source(lake) rule predicts a lake as the best water source if the location is far from rivers but close to a lake.
The first best_water_source(river) rule predicts a river as the best water source if the location is close to a river and has low rainfall.
The second best_water_source(river) rule predicts a river as the best water source if the location is close to a river, has medium rainfall, and the aquifer is sandy.
The second best_water_source(rain) rule predicts rain as the best water source if the location is close to a river, has medium rainfall, and the aquifer is not sandy.
The last best_water_source(rain) rule predicts rain as the best water source if the location is far from rivers and lakes and has high rainfall.
Finally, the main rule is defined. It is the main goal that is called to predict the best water source based on the location factors. It first displays a message to indicate that it is predicting the best water source. Then, it calls the best_water_source rule with a variable WaterSource to store the predicted water source. Finally, it displays the predicted water source to the user.

To use this code, consult the file in a Prolog interpreter and then call the main rule to see the predicted water source based on the provided location factors.