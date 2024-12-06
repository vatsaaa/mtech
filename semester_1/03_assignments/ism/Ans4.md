# Question 4
In a comparison of the cleaning action of four detergents, 20 pieces of white cloth were
first soiled with India ink. The clothes were then washed under controlled conditions with 5
pieces washed by each of the detergents. Unfortunately, three pieces of cloth were ‘lost’ in
the course of the experiment. Whiteness readings, made on the 17 remaining pieces of cloth
are given in the JSON below:
[
        {
            "Detergent": "A",
            "Clothes Count": "5",
            "Whiteness Readings": [77, 81, 61, 76, 69]
        },
        {
            "Detergent": "B",
            "Clothes Count": "3",
            "Whiteness Readings": [74, 66, 58]
        },
        {
            "Detergent": "C",
            "Clothes Count": "5",
            "Whiteness Readings": [73, 78, 57, 69, 63]
        },
        {
            "Detergent": "D",
            "Clothes Count": "4",
            "Whiteness Readings": [76, 85, 77, 64]
        }
    ]

Assuming all whiteness readings to be normally distributed with common variance,
test the hypothesis of no difference between the four brands as regards mean
whiteness readings after washing at 5% LOS.


# Answer 4
To test the hypothesis that after washing there is no difference between mean whiteness readings 
of the four brands, we use Analysis of Variance (ANOVA) as there are multiple groups to compare.


## Step 1: Setup the Hypotheses
Null Hypothesis (H₀): There is no significant difference in mean whiteness readings among the four detergents.
H₀: μ₁ = μ₂ = μ₃ = μ₄

Alternative Hypothesis (H₁): There is a significant difference in mean whiteness readings among the four detergents.
H₁: At least one of the means (μ₁, μ₂, μ₃, μ₄) is different


## Step 2: Degrees of Freedom:
Calculate the degrees of freedom for between-groups (df-between) and within-groups (df-within)

Degrees of Freedom Between (df-between):
df-between = k − 1
k: the number of groups (i.e. detergents in this case).

∴ k = 4
df-between = 4 − 1
df-between = 3


Degrees of Freedom Within (df-within):

N: the total number of observations
Since 3 clothes were lost, here total number of observations is 17 (and not 20)
∴ N = 17

k: number of groups
k = 4

df-within = N − k
df-within = 17 - 4
df-within = 13


## Step 3: Calculate F-Statistic:
F = MeanSquare-between/MeanSquare-within
nᵢ: number of observations in group 
X-barᵢ: mean of the group 
X-bar-grand: overall mean

                        k       nᵢ
X-bar-grand = (1/N) * ( ∑       ∑     Xᵢⱼ)
                        ᵢ₌₁     ⱼ₌₁

=> X-bar-grand = (77+81+61+76+69+74+66+58+73+78+57+69+63+76+85+77+64)/17
=> X-bar-grand = 70.8236

X-bar-A = (77+81+61+76+69)/5 = 72.8
X-bar-B = (74+66+58)/3 = 66
X-bar-C = (73+78+57+69+63)/5 = 68
X-bar-D = (76+85+77+64)/4 = 75.5

      k
​SSB = ∑ nᵢ (X-barᵢ - X-bar-grand)²
      ᵢ₌₁


SSB-A = 19.5321683045
SSB-B = 23.2664416609
SSB-C = 39.8616083045
SSB-D = 87.4774866436

SSB = SSB-A + SSB-B + SSB-C + SSB-D
SSB = (19.5321683045+23.2664416609+39.8616083045+87.4774866436) 
SSB = 170.1377049135


Sum of Squares Within (SSW):
      k       nᵢ
SSW = ∑       ∑     (Xᵢⱼ - X-barᵢ)²
      ᵢ₌₁     ⱼ₌₁


SSW-A​ = (77−72.8)² + (81−72.8)² + (61−72.8)² + (76−72.8)² + (69−72.8)²
SSW-A​ = (4.2)² + (8.2)² + (−11.8)² + (3.2)² + (−3.8)²
SSW-A​ = 248.8

SSW-B​ = (74 − 66)² + (66 − 66)² + (58−66)2
SSW-B​ = 8² + 0² + (−8)²
SSW-B = 128

SSW-C​ = (73 − 68)² + (78 − 68)² + (57 − 68)² + (69 − 68)² +(63 − 68)²
SSW-C​ = 5² + 10 + (−11)² + 1² + (−5)²
SSW-C = 272


SSW-D = (76 − 75.5)² + (85 − 75.5)²+(77 − 75.5)²+(64 − 75.5)²
SSW-D = (0.5)² + (9.5)² + (1.5)² + (−11.5)²
SSW-D = 225

SSW = SSW-A + SSW-B​ + SSW-C + SSW-D
​SSW = 248.8 + 128 + 272 + 225
SSW = 873.8

where:
Xᵢⱼ: j-th observation in group i
X-barᵢ: mean of group i

MSB = Sum of Squares Between (SSB) / Degrees of Freedom Between (df-between)
MSW = Sum of Squares Within (SSW) / Degrees of Freedom Within (df-within)


MSW = (SSW/df-within)
​MSW = (873.8 / 13)
​MSW = 67.2153846154
​
F = (MeanSquare-between/MeanSquare-within)
F = (170.1377/3)/(873.8/13)
F = 0.84374



## Step 4: Critical Value:
Determine the critical F-value from the F-distribution table for the chosen level of significance
(5% in this case) and degrees of freedom. 


## Step 5: Perform ANOVA:
Use the whiteness readings data for each detergent to calculate the ANOVA statistic.
The ANOVA test compares the variation within each group to the variation between groups.


## Step 6: Decision:
Compare the calculated F-statistic with the critical F-value.
If the calculated F-statistic is greater than the critical value, reject the null hypothesis.

