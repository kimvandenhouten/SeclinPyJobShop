import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

fitting = False

# Load data
df = pd.read_csv("factory_data/fermentation_times/ferm_times.csv", delimiter=";")

# Convert 'tferm (hrs)' to float
df['tferm (hrs)'] = df['tferm (hrs)'].astype(str).str.replace(',', '.', regex=False).astype(float)

# We now merge the ferm times that were execution on V200 and V218 because they come from the same recipe
df['Machine'] = df['Machine'].replace({'F200': 'F218'})


# Create Machine_PROD column
df['Machine_PROD'] = df['Machine'] + ' - ' + df['PROD']

# Count the occurrences of each unique Machine_PROD value
counts = df['Machine_PROD'].value_counts()

# Display the counts
print(counts)

# Get statistics on the counts
print("\nStatistics on the counts:")
print(f"Average: {counts.mean():.2f}")
print(f"Min: {counts.min()}")
print(f"Max: {counts.max()}")
print(f"Std: {counts.std():.2f}")
print(f"Total unique combinations: {counts.count()}")

# Some stats about the times
ferm_stats = df.groupby('Machine_PROD')['tferm (hrs)'].agg(['count', 'min', 'mean', 'max', 'std'])
print(ferm_stats)

if fitting:
    # === CONFIGURATION ===
    # Choose the Machine_PROD combination you want to analyze
    for target_combination in df['Machine_PROD'].unique():

        # Filter data
        data = df[df['Machine_PROD'] == target_combination]['tferm (hrs)'].dropna()

        print(data)

        # === CANDIDATE DISTRIBUTIONS ===
        distributions = {
            'norm': stats.norm,
            'lognorm': stats.lognorm,
            'expon': stats.expon,
            'gamma': stats.gamma,
            'weibull_min': stats.weibull_min
        }

        results = []

        # === FIT AND EVALUATE ===
        for name, dist in distributions.items():
            try:
                # Fit distribution
                params = dist.fit(data)

                # Calculate log-likelihood
                log_likelihood = np.sum(dist.logpdf(data, *params))

                # Number of parameters
                k = len(params)

                # AIC = 2k - 2ln(L)
                aic = 2 * k - 2 * log_likelihood

                # KS test
                ks_stat, ks_p = stats.kstest(data, name, args=params)

                results.append({
                    'distribution': name,
                    'params': params,
                    'aic': aic,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p
                })
            except Exception as e:
                print(f"Error fitting {name}: {e}")

        # === RANK RESULTS ===
        results = sorted(results, key=lambda x: x['aic'])
        best = results[0]

        print("Best fit based on AIC:")
        for r in results:
            print(f"{r['distribution']:12s} | AIC: {r['aic']:.2f}, KS stat: {r['ks_stat']:.3f}, p = {r['ks_p']:.3f}")

        # === PLOT BEST FIT ===
        dist = distributions[best['distribution']]
        params = best['params']

        x = np.linspace(min(data), max(data), 100)
        pdf_fitted = dist.pdf(x, *params)

        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=30, stat='density', color='skyblue', label='Empirical')
        plt.plot(x, pdf_fitted, 'r-', label=f'Best Fit: {best["distribution"]}')
        plt.title(f'Best Distribution Fit for {target_combination}')
        plt.xlabel('tferm (hrs)')
        plt.ylabel('Density')
        plt.legend()
        plt.ylim(top=1)
        plt.tight_layout()
        plt.savefig(f'factory_data/fermentation_times/fitted_distributions/{target_combination}.png')

        # === PLOT ALL FITTED DISTRIBUTIONS ===
        x = np.linspace(min(data), max(data), 100)

        plt.figure(figsize=(12, 7))
        sns.histplot(data, bins=30, stat='density', color='lightgrey', label='Empirical', edgecolor='black')

        for r in results:
            dist_name = r['distribution']
            dist = distributions[dist_name]
            params = r['params']

            try:
                pdf = dist.pdf(x, *params)
                plt.plot(x, pdf, label=f"{dist_name} (AIC: {r['aic']:.1f})")
            except Exception as e:
                print(f"Error plotting {dist_name}: {e}")

        plt.title(f'All Fitted Distributions for {target_combination}')
        plt.xlabel('tferm (hrs)')
        plt.ylabel('Density')
        plt.legend()
        plt.ylim(top=1)
        plt.tight_layout()
        plt.savefig(f'factory_data/fermentation_times/fitted_distributions/{target_combination}_all_fits.png')

        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=30, stat='count', color='skyblue', label='Empirical')
        plt.title(f'Empirical Distribution Fit for {target_combination}, number of samples {len(data)} mean {np.round(np.mean(data), 1)}')
        plt.xlabel('tferm (hrs)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'factory_data/fermentation_times/fitted_distributions/{target_combination}_empirical.png')
