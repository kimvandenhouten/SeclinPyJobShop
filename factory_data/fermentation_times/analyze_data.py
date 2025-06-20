import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example: load your data
df = pd.read_csv("your_data.csv")  # assuming your data is in CSV

# Create a new column for the Machine–PROD combination
df['Machine_PROD'] = df['Machine'] + ' - ' + df['PROD']

# Set the size of the plot
plt.figure(figsize=(14, 8))

# Use seaborn to plot histograms per Machine–PROD combination
sns.histplot(data=df, x='tferm (hrs)', hue='Machine_PROD', multiple='stack', bins=30)

plt.title('Histogram of tferm (hrs) per Machine–PROD combination')
plt.xlabel('tferm (hrs)')
plt.ylabel('Count')
plt.legend(title='Machine - PROD', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()