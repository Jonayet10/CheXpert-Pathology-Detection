import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('train.csv')

# Display basic info and first few rows to understand the data structure
print(df.info())
print(df.head())

# Male to Female Ratio
sex_counts = df['Sex'].value_counts()
print("Male to Female Ratio:", sex_counts['Male'] / sex_counts['Female'])

# Plotting the sex distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=df)
plt.title('Distribution of Sex')
plt.savefig('sex_distribution.png')  # Saves the plot as a PNG file
plt.close()  # Closes the figure to free up memory

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png')
plt.close()

# Pathologies Distribution
pathologies = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 
               'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

# Prepare data for plotting
pathology_counts = {pathology: [df[df[pathology] == 1][pathology].count(), 
                                df[df[pathology] == -1][pathology].count()] for pathology in pathologies}

positive_counts = [pathology_counts[path][0] for path in pathologies]
uncertain_counts = [pathology_counts[path][1] for path in pathologies]

# Plotting each pathology count
plt.figure(figsize=(14, 8))
bar_width = 0.4  # width of bars

indices = range(len(pathologies))  # the label locations
plt.bar(indices, positive_counts, width=bar_width, label='Positive')
plt.bar([i + bar_width for i in indices], uncertain_counts, width=bar_width, label='Uncertain')

plt.title('Distribution of Pathologies')
plt.xlabel('Pathology')
plt.ylabel('Count')
plt.xticks([i + bar_width / 2 for i in indices], pathologies, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('pathologies_distribution.png')
plt.close()
