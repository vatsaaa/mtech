# Question 1
A company has the head office at Chennai and a branch at Coimbatore. The personnel director wanted
to know if the workers at the two places would like the introduction of a new plan of work and a 
survey was conducted for this purpose. Out of a sample of 500 workers at Chennai, 62% favoured the
new plan. At Coimbatore out of a sample of 400 workers, 41% were against the new plan. Is there any
significant difference between the two groups in their attitude towards the new plan at 5% level?
Elaborate how to make the conclusion.

# Answer 1
Determining if there is a significant difference between the attitude of Chennai workers and Coimbatore workers towards the new plan requires a hypothesis test. 

## Step 1: Setup the Hypotheses
We are comparing the proportion of workers favoring the new plan in Chennai to the proportion of workers favoring the new plan in Coimbatore.

n1: Sample size of workers in Chennai 
=> n1 = 500

p1: Proportion of workers in Chennai favoring the new plan
∴ p̂1 = 0.62

x1 = Number of workers in Chennai favoring the new plan
∴ x1 = n * p̂1
=> x1 = 0.62×500
=> x1 = 310

n2: Sample size of workers in Coimbatore 
=> n2 = 400

p2: Proportion of workers in Coimbatore favoring the new plan
∴ p̂2 = 1 - Proportion of Coimbatore workers not favoring the new plan
=> p̂2 = 1 - 0.41
=> p̂2 = 0.59

x2 = Number of workers in Coimbatore favoring the new plan
∴ x2 = n * p̂2
=> x2 = 0.59 × 400
x2 = 236

Conditions of normality are met as:
n1 * p̂1 > 5
n1 * (1 − p̂1) > 5
and 
n2 * p̂2 > 5
n2 * (1 − p̂2) > 5

<ins>We will use a z-test for comparing two proportions.</ins>

<b>Null hypothesis (H0):</b> There is no significant difference between the attitudes of the workers at Chennai and the workers at Coimbatore
H0: p1 = p2

<b>Alternative hypothesis (H1):</b> There is a significant difference between the attitudes of the workers at Chennai and the workers at Coimbatore
H1: p1 ≠ p2

Note, this will be a two-tailed z-test.


## Step 2: Decide the significance level (⍺)
LOS(⍺): Level of significance is given as 5%
=> ⍺ = 0.05


## Step 3: Calculate test statistic
Using a two-tailed z-test for comparing two proportions to test these hypotheses. 
The formula for the z test statistic is:
z = (p̂1 - p̂2)/SE

p̂ = p: Pooled sample proportion 
p = (x1 +x2)/(n1​ + n2)
p = (x1+x2)/(n1+n2) = (310 + 236)/(500 + 400) = 0.60667

SE = sqrt( p̂ * (1 - p̂) * ( (1/n1) + (1/n2) ) )
SE = sqrt(0.60667 * (1 - 0.60667) * ((1/400) + (1/500)))
SE = 0.03277

z = (p̂1 - p̂2)/SE
=> z = (0.62 − 0.59)/0.03277 = 0.91550


## Step 4: Make a decision about null hypotheses
To make a decision we need to compare the calculated z-value to the critical value of the standard normal distribution to make a decision.

Since this is a two-tailed test, Critical z-value = ±1.96

Since, ∣0.91550∣ < 1.96
=> Calculated z-value does not fall in the rejection region
∴ Decision: fail to reject the null hypothesis (H0) 

## Step 5: Conclusion
=> There is not enough evidence to suggest a significant difference in attitudes, towards the new plan, between the workers in Chennai and Coimbatore, at the 5% significance level. Note that failing to reject the null hypothesis does not imply the groups attitudes are exactly the same. 

