# Question 3
In 120 throws of a single die, the distribution of faces observed is given in the JSON: 

[
    {
        "Face": "1",
        "Frequency": "30"
    },
    {
        "Face": "2",
        "Frequency": "25"
    },
    {
        "Face": "3",
        "Frequency": "18"
    },
    {
        "Face": "4",
        "Frequency": "10"
    },
    {
        "Face": "5",
        "Frequency": "22"
    },
    {
        "Face": "6",
        "Frequency": "15"
    }
]

Taking a level of significance as 5%, please explain how can we determine if the dice is biased.

# Answer 3
To determine if the die is biased, we need to perform a chi-square goodness-of-fit test. 
Since the chi-square test compares the observed frequencies with the expected frequencies under
the assumption that the die is fair (i.e., each face has an equal probability of 1/6).

## Step 1: Setup the Hypotheses
Null Hypothesis (H₀): The die is fair (not biased).
Alternative Hypothesis (H₁): The die is biased.

Under the assumption of a fair die, each face should occur with a probability of 1/6 and we 
expect each face to appear 20 times in 120 throws.

∴ Each face has an equal probability of 1/6 and the total number of throws in this case is 120.

Eᵢ = Total throws/Number of faces
Eᵢ = 120/6 = 20

## Step 2: Decide the significance level (⍺)
LOS(⍺): Level of significance is given as 5%
∴ ⍺ = 0.05

## Step 3: Calculate the test statistic
For each face, we find the squared difference between the observed and expected frequencies, divided by the expected frequency. Sum of each of these values gives us the chi-square statistic.

Oᵢ: Observed frequency for face #i | i ϵ [1, 6]
Eᵢ: Expected frequency for face #i | i ϵ [1, 6]


χ² = ∑ ((Oᵢ − Eᵢ)²/Eᵢ)


χ² = ((30−20)²/20) + ((25−20)²/20) + ((18−20)²/20) + ((10−20)²/20) + ((22−20)²/20) + ((15−20)²/20)
χ² = 12.9

## Step 4. Degrees of Freedom (df): is equal to the number of categories (faces) minus 1
df = 6 - 1
df = 5

Determine the critical value from the chi-square distribution table for the chosen
level of significance (5% in this case) and the degrees of freedom.

From chi-square distribution table,
Critical-χ² = 11.0705

Calculated chi-square statistic > Critical value of chi-square statistic
=> Null hypothesis is rejected


## Step 5: Conclusion
For 120 throws of our dice, 
Calculated-χ² (12.9000) > Critical-χ² (11.0705); at a 5% level of significance with 5 degrees of freedom

Thus we reject the null hypothesis (H₀), concluding that there is evidence to conclude that the die is biased.

