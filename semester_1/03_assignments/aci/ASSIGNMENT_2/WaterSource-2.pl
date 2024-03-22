predict_water_source(Distance_from_Lake, Distance_from_River, Rainfall_intensity, Sandy_aquifer, Distance_from_Beach, WaterSource) :-
(
    Distance_from_Lake < 10,
    WaterSource = lake
);
(
    Distance_from_Lake >= 10,
    Distance_from_River < 8,
    Rainfall_intensity < 200,
    WaterSource = river
);
(
    Distance_from_Lake >= 10,
    Distance_from_River < 8,
    Rainfall_intensity >= 200,
    WaterSource = rain
);
(
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall_intensity >= 150,
    WaterSource = rain
);
(
	Distance_from_Lake < 14,
	Distance_from_River >= 8,
	Rainfall_intensity < 150,
	Sandy_aquifer == no,
	WaterSource = lake
);
(
	Distance_from_Lake >= 14,
	Distance_from_River >= 8,
	Rainfall_intensity < 150,
	Sandy_aquifer == no,
	WaterSource = rain
);
(
	Distance_from_Lake >= 10,
	Distance_from_River >= 8,
	Rainfall_intensity < 150,
	Sandy_aquifer == yes,
    Distance_from_Beach >= 5,
	WaterSource = groundwater
);
(
    Distance_from_Lake >= 10,
    Distance_from_River >= 20,
    Rainfall_intensity < 150,
    Sandy_aquifer == yes,
    Distance_from_Beach < 5,
    WaterSource = rain
);
(
    Distance_from_Lake >= 10,
    Distance_from_River < 20,
    Rainfall_intensity < 150,
    Sandy_aquifer == yes,
    Distance_from_Beach < 5,
    WaterSource = river
);
% default to groundwater
WaterSource = groundwater.

get_user_input :-
    write('Distance from Lake (km): '),
    read(Distance_from_Lake),
    write('Distance from River (km): '),
    read(Distance_from_River),
    write('Enter Rainfall_intensity (mm/month): '),
    read(Rainfall_intensity),
    write('Is the Sandy_aquifer Sandy? (yes/no): '),
    read(Sandy_aquifer),
    write('Distance from Beach (km): '),
    read(Distance_from_Beach),
    predict_water_source(Distance_from_Lake, Distance_from_River, Rainfall_intensity, Sandy_aquifer, Distance_from_Beach, WaterSource),
    write('Recommended Water Source: '),
    write(WaterSource).

% Start the program
start :-
    write('Welcome to the Water Source Prediction System!'), nl,
    get_user_input,
    nl,
    write('Thank you!').