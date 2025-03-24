import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


class ConditionalKernelDensityEstimator(BaseEstimator):

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_models = {}

    def fit(self, X, y):
        """
        Fit the Conditional Kernel Density Estimation model.

        Parameters:
        - X: DataFrame of categorical variables.
        - y: Array-like target variable (continuous or discrete).
        """
        # Convert X to a tuple of unique category combinations
        X_tuples = X.apply(tuple, axis=1)

        # Find all unique category combinations
        unique_conditions = X_tuples.unique()

        # Fit a separate KDE model for the target variable for each condition
        for condition in unique_conditions:
            mask = X_tuples == condition
            y_subset = y[mask]

            if len(y_subset) > 0:  # Ensure there's enough data for KDE
                kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
                if isinstance(y_subset, pd.Series):
                    y_subset = y_subset.values
                kde.fit(y_subset.reshape(-1, 1))  # KDE operates on numerical y
                self.kde_models[condition] = kde

    def score_samples(self, X, y_values):
        """
        Compute the log density of y given X.

        Parameters:
        - X: DataFrame of categorical variables.
        - y_values: Target values to evaluate.

        Returns:
        - log densities: A dictionary mapping X conditions to log densities.
        """
        X_tuples = X.apply(tuple, axis=1)
        log_densities = {}

        for condition, y_val in zip(X_tuples, y_values):
            if condition in self.kde_models:
                kde = self.kde_models[condition]
                log_density = kde.score_samples(np.array([[y_val]]))

                log_densities[condition] = log_density[0]
            else:
                log_densities[condition] = None  # No KDE model for this condition

        return log_densities

    def sample(self, n_samples, X_condition):
        """
        Sample from the conditional density of the target variable given X.

        Parameters:
        - n_samples: Number of samples to generate.
        - X_condition: A tuple representing a specific category combination.

        Returns:
        - samples: Generated samples from the conditional density.
        """
        X_condition = tuple(X_condition)
        if X_condition not in self.kde_models:
            raise ValueError(f"No KDE model found for condition {X_condition}")

        kde = self.kde_models[X_condition]
        return kde.sample(n_samples).flatten()

    def plot_kde_with_histogram(self, X_condition, X_data, y_data, bins=100):
        """
        Plot KDE and histogram for the target variable conditioned on X.

        Parameters:
        - X_condition: A tuple representing a specific category combination.
        - y_data: The actual target variable values for this condition.
        - bins: Number of bins in the histogram.
        """

        if X_condition not in self.kde_models:
            print(f"No KDE model found for condition {X_condition}")
            return

        X_tuples = X_data.apply(tuple, axis=1)
        mask = X_tuples == X_condition

        y_subset = y_data[mask]

        kde = self.kde_models[X_condition]

        # Generate a range of y values for plotting the KDE curve
        y_min, y_max = min(y_subset), max(y_subset)
        y_range = np.linspace(y_min, y_max, 100).reshape(-1, 1)
        kde_estimates = np.exp(kde.score_samples(y_range))  # Convert log density to density

        # Plot the histogram and KDE
        plt.figure(figsize=(8, 5))
        sns.histplot(y_subset, bins=bins, kde=False, stat="density", color="lightblue", label="Histogram")
        plt.plot(y_range, kde_estimates, color="red", label="KDE Estimate")
        plt.xlabel("Target Variable (y)")
        plt.ylabel("Density")
        plt.title(f"KDE and Histogram for {X_condition}")
        plt.legend()
        plt.show()

#
# data = df[cat_cols][:4000]
# target = df['ServTime'][:4000] / 60
#
# # Create and fit the CKDE model
# ckde = ConditionalKernelDensityEstimator(bandwidth=0.2)
# ckde.fit(data, target)


class DurationModel:

    def __init__(self, model=ConditionalKernelDensityEstimator):
        self.model = model()

    def train(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.model.fit(X, y)

    def predict(self, X: tuple | list | np.ndarray) -> np.ndarray:
        return self.model.sample(1000, X)

