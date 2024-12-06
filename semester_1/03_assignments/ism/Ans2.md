# Question 2
A soap manufacturing company was distributing a particular brand of soap through a large
number of retail shops. Before a heavy advertisement campaign, the mean sales per week per
shop was 140 dozen. After the campaign a sample of 26 shops was taken and the mean sales
was found to be 147 dozen with a standard deviation of 16 dozen. Is the advertisement
effective? Take level of significance (LOS) as 5%.


# Answer 2
To determine whether the advertisement is effective, we need to perform a hypothesis test. 
We use one-sample t-test to compare the mean of a sample to a known value (i.e. the mean
before the advertisement campaign) and determine if the difference is statistically significant.

## Step 1: Setup the Hypotheses
We define the null and alternative hypotheses as following:

Null Hypothesis (H0): The advertisement is not effective. Hence, the mean sales per week per shop
after the campaign (μ) remains the same as was before the campaign (μ0​).
H0: μ = μ₀

Alternative Hypothesis (H1): The advertisement is effective, and the mean sales per week per shop
after the campaign (μ) is greater than the mean that was before the campaign (μ0).
H1: μ > μ₀

Mean sales before the advertisement campaign: μ₀ = 140
Sample mean sale after the advertisement campaign: x-bar = 147
Sample size: n = 26
Sample standard deviation (s) = 16 dozen
Level of Significance (LOS) = 5%


## Step 2: Decide the significance level (⍺)
LOS is the probability of making a Type I error, i.e. error of rejecting a true null hypothesis.
∴ ⍺= 0.05

We will use a one-sample t-test for this analysis since the population standard deviation is not
known and the sample size (n) is relatively small i.e. n < 30.

## Step 3: Calculate the test statistic
t = (x-bar - μ₀)/(s/√n)
=> t = (147−140)/(16/√26)
=> t = 2.231

Degrees of freedom: df = n - 1 = 26 - 1 = 25
Critical t-value: t(⍺, df)
​
Using t-table for a one-tailed test, with df=25 and ⍺ = 0.05 we find the critical t-value
t(⍺, df) = 1.708

## Step 4: Make a decision about null hypothesis
2.231 > 1.708
=> Calculated t-value > Critical t(⍺, df)

## Step 5: Conclusion
We reject the null hypothesis at the 5% significance level. This suggests that based on the
results of the one-tailed t-test, we have statistical evidence to support the effectiveness
of the advertisement campaign in increasing the mean sales per week per shop.


