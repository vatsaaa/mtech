predict_water_source(LakeDistance, RiverDistance, Rainfall, Aquifer, BeachDistance, WaterSource) :-
(
    LakeDistance < 10,
    WaterSource = lake
);
(
    LakeDistance >= 10,
    RiverDistance < 8,
    Rainfall < 200,
    WaterSource = river
);
(
    LakeDistance >= 10,
    RiverDistance < 8,
    Rainfall >= 200,
    WaterSource = rain
);
(
    LakeDistance >= 10,
    RiverDistance >= 8,
    Rainfall >= 150,
    WaterSource = rain
);
(
	LakeDistance > 10,
	LakeDistance < 14,
	RiverDistance >= 8,
	Rainfall < 150,
	Aquifer == not_sandy,
	WaterSource = lake
);
(
	LakeDistance >= 14,
	RiverDistance >= 8,
	Rainfall < 150,
	Aquifer == not_sandy,
	WaterSource = rain
);
(
	LakeDistance >= 10,
	RiverDistance >= 8,
	Rainfall < 150,
	Aquifer == sandy,
	WaterSource = groundwater
);
(
    LakeDistance >= 14,
    RiverDistance >= 20,
    Rainfall < 150,
    Aquifer == sandy,
    BeachDistance < 5,
    WaterSource = rain
);
(
    LakeDistance >= 14,
    RiverDistance >= 8,
    Rainfall < 20,
    Rainfall < 150,
    Aquifer == sandy,
    BeachDistance < 5,
    WaterSource = river
);
% default to groundwater
WaterSource = groundwater.

get_user_input :-
    write('Distance from Lake (km): '),
    read(LakeDistance),
    write('Distance from River (km): '),
    read(RiverDistance),
    write('Enter Rainfall (mm/month): '),
    read(Rainfall),
    write('Is the Aquifer Sandy? (yes/no): '),
    read(Answer),
    (Answer == yes -> Aquifer = sandy; Aquifer = not_sandy),
    write('Distance from Beach (km): '),
    read(BeachDistance),
    predict_water_source(LakeDistance, RiverDistance, Rainfall, Aquifer, BeachDistance, WaterSource),
    write('Recommended Water Source: '),
    write(WaterSource).

% Start the program
start :-
    write('Welcome to the Water Source Prediction System!'), nl,
    get_user_input,
    nl,
    write('Thank you!').