lakeDistance(9).
riverDistance(21).
rainfall(149).
sandyAcquifer(yes).
beachDistance(4).

% Rule for determining water resource based on various factors
waterResource(Decision) :-
    lakeDistance(DistLake), DistLake < 10,
    Decision = 'The best water source for the community is: Lake'.
waterResource(Decision) :-
    lakeDistance(DistLake1), DistLake1 >= 10, DistLake1 < 14,
    riverDistance(RiverDist), RiverDist >= 8,
    rainfall(Rainfall), Rainfall < 150,
    sandyAcquifer(Sandy), Sandy = 'No',
    Decision = 'The best water source for the community is: Lake'.
waterResource(Decision) :-
    lakeDistance(DistLake2), DistLake2 >= 10,
    riverDistance(RiverDist2), RiverDist2 >= 8,
    rainfall(Rainfall2), Rainfall2 < 150,
    sandyAcquifer(Sandy), Sandy = 'Yes',
    beachDistance(BeachDist), BeachDist >= 5,
    Decision = 'The best water source for the community is: GroundWater'.
waterResource(Decision) :-
    lakeDistance(DistLake3), DistLake3 >= 10,
    riverDistance(RiverDist3), RiverDist3 < 8,
    rainfall(Rainfall3), Rainfall3 < 200,
    Decision = 'The best water source for the community is: River'.
waterResource(Decision) :-
    lakeDistance(DistLake4), DistLake4 >= 10,
    riverDistance(RiverDist4), RiverDist4 >= 8, RiverDist4 < 20,
    rainfall(Rainfall4), Rainfall4 < 150,
    sandyAcquifer(Sandy2), Sandy2 = yes,
    beachDistance(BeachDist2), BeachDist2 < 5,
    Decision = 'The best water source for the community is: River'.
waterResource(Decision) :-
    lakeDistance(DistLake5), DistLake5 >= 10,
    riverDistance(RiverDist5), RiverDist5 < 8,
    rainfall(Rainfall5), Rainfall5 >= 200,
    Decision = 'The best water source for the community is: Rain'.
waterResource(Decision) :-
    lakeDistance(DistLake6), DistLake6 >= 10,
    riverDistance(RiverDist6), RiverDist6 >= 8,
    rainfall(Rainfall6), Rainfall6 >= 150,
    Decision = 'The best water source for the community is: rain'.
waterResource(Decision) :-
    lakeDistance(DistLake7), DistLake7 >= 14,
    riverDistance(RiverDist7), RiverDist7 >= 8,
    sandyAcquifer(Sandy2), Sandy2 = no,
    rainfall(Rainfall7), Rainfall7 < 150,
    Decision = 'rain'.
waterResource(Decision) :-
    lakeDistance(DistLake8), DistLake8 >= 10,
    riverDistance(RiverDist8), RiverDist8 >= 20,
    beachDistance(BeachDist2), BeachDist2 < 5,
    sandyAcquifer(Sandy2), Sandy2 = yes,
    rainfall(Rainfall8), Rainfall8 < 150,
    Decision = 'The best water source for the community is: Rain'.
waterResource(Decision) :-
    % Default case if none of the rules match
    Decision = 'The best water source for the community is: unknown'.
