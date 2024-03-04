/* Rules to determine the proximity to water sources */
close_to_river(Distance) :- 
  write('How far is the location from the nearest river (in Km) ? '),
  read(Distance),
  Distance < 20.

close_to_lake(Distance) :- 
  write('How far is the location from the nearest lake (in Km) ? '),
  read(Distance),
  Distance < 14.

close_to_beach(Distance) :- 
  write('How far is the location from the nearest beach (in Km) ? '),
  read(Distance),
  Distance < 5.

/* Rules to determine the rainfall intensity */
low_rainfall :- 
  write('What is the average monthly rainfall amount (in mm) ? '),
  read(Rainfall),
  Rainfall < 150.

medium_rainfall :- 
  write('What is the average monthly rainfall amount (in mm) ? '),
  read(Rainfall),
  Rainfall >= 150, Rainfall < 200.

high_rainfall :- 
  write('What is the average monthly rainfall amount (in mm) ? '),
  read(Rainfall),
  Rainfall >= 200.

/* Rule to determine the aquifer type */
sandy_aquifer :- 
  write('Is the aquifer sandy (yes/no) ? '),
  read(Aquifer),
  Aquifer == yes.

/* Rules to predict the water source based on the decision tree */
best_water_source(groundwater) :-
  close_to_river(Distance), Distance >= 20,
  close_to_lake(Distance), Distance >= 14,
  not(close_to_beach(_)).

best_water_source(lake) :-
  close_to_river(Distance), Distance >= 20,
  close_to_lake(Distance), Distance < 14.

best_water_source(river) :-
  close_to_river(Distance), Distance < 20,
  low_rainfall.

best_water_source(river) :-
  close_to_river(Distance), Distance < 20,
  medium_rainfall,
  sandy_aquifer.

best_water_source(rain) :-
  close_to_river(Distance), Distance < 20,
  medium_rainfall,
  not(sandy_aquifer).

best_water_source(rain) :-
  close_to_river(Distance), Distance >= 20,
  high_rainfall.

/* Main goal to call the decision tree rules */
main :-
  write('Predicting the best water source based on location factors:'), nl,
  best_water_source(WaterSource),
  write('The best water source for the community is: '),
  write(WaterSource), nl.