import monte_carlo_pi
import matplotlib
from matplotlib import pyplot as plt
import math
print("Enter the desired number of trials for π estimation:")
num_trials = input()
num_trials = int(num_trials)
# here, we'll run the monte carlo sim as many times as specified by the user, 
# and store the results in the dictionary "estimates"
estimates = []
for n in range(num_trials):
    pi_estimate, x, y, inside = monte_carlo_pi.estimate_pi(10000, seed=None)
    estimates.append(pi_estimate)
# Now we can analyze through quite a few statistical methods, we'll intuitively start first with mean
# then move on to variance, standard deviation, and finally confidence intervals.
mean_estimate = sum(estimates) / len(estimates)
print(f"Mean of π estimates after {num_trials} trials: {mean_estimate}")
squared_diffs = [(est - mean_estimate) ** 2 for est in estimates]
standard_deviation = (sum(squared_diffs) / len(estimates)) ** 0.5
print(standard_deviation)
#next, add analysis scaffold ( or warmup samp), use z score filter and stochastic confidence bands
estimates_run = []
for n in range(0, 300):
    estimates_run.append(estimates[n])
for n in range(300, num_trials):
    current_mean = sum(estimates_run) / len(estimates_run)
    current_std = (sum([(est - current_mean) ** 2 for est in estimates_run]) / len(estimates_run)) ** 0.5
    z_score = (estimates[n] - current_mean) / current_std if current_std != 0 else 0
    if abs(z_score) < 2.5:  # Using a z-score threshold of 2.5 for filtering
        estimates_run.append(estimates[n])
#clc stcstc bands
means = []
upper = []
lower = []
s = 0.0
s2 = 0.0
for i, e in enumerate(estimates_run, start=1):
    s += e
    s2 += e * e

    mean = s / i
    var = (s2 / i) - mean**2
    std = math.sqrt(var) if var > 0 else 0.0
    se = std / math.sqrt(i)

    means.append(mean)
    upper.append(mean + 1.96 * se)
    lower.append(mean - 1.96 * se)
# stochastic confidence bands computed, now we can plot the running mean of the filtered estimates 
running_means = []
s = 0.0

for i, est in enumerate(estimates_run, start=1):
    s += est
    running_means.append(s / i)
# Finally, we can plot the running mean of the π estimates to visualize convergence
plt.plot(means, label="Running mean (filtered)")
plt.fill_between(
    range(len(means)),
    lower,
    upper,
    alpha=0.25,
    label="95% confidence band"
)
cut = 5000
pi_hat = sum(means[cut:]) / len(means[cut:])
print(f"Final π estimate after filtering and convergence (from sample {cut} onwards): {pi_hat}")
plt.axhline(3.141592653589793, linestyle="--", label="True π")
plt.axhline(pi_hat, linestyle="--", label="filtered pi mean", color="green")
plt.axhline(mean_estimate, linestyle="--", label="mean of unfiltered estimates",color = "orange")
plt.xlabel("Accepted samples (after filter)")
plt.ylabel("π estimate")
plt.title(f"Filtered Monte Carlo π with Confidence Envelope\nFinal π estimate after filtering and convergence (from sample {cut} onwards): {pi_hat}")
plt.legend()
plt.show()