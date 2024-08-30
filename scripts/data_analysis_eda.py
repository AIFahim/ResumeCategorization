import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os

# Define paths
data_dir = 'data'
plots_dir = 'plots'
resume_csv_path = os.path.join(data_dir, 'Resume.csv')

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
resume_data = pd.read_csv(resume_csv_path)

# 1. Analyze Category Distribution
category_distribution = resume_data['Category'].value_counts()

# Plot the distribution
plt.figure(figsize=(10, 6))
category_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Resume Categories')
plt.xlabel('Category')
plt.ylabel('Number of Resumes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'category_distribution.png'))
plt.show()

# 2. Analyze Text Length (word count)
resume_data['word_count'] = resume_data['Resume_str'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 6))
sns.histplot(resume_data['word_count'], kde=True, color='purple')
plt.title('Distribution of Word Counts in Resumes')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'word_count_distribution.png'))
plt.show()

# 3. Generate a Word Cloud for Resume Text
text = " ".join(resume_data['Resume_str'].values)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Resume Text')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'word_cloud.png'))
plt.show()

# Output basic statistics
print("Basic statistics of the resume dataset:")
print(resume_data.describe())

# Output information about the dataset
print("\nDataset Info:")
print(resume_data.info())

print(f"Data analysis and EDA complete. Visualizations saved in '{plots_dir}' directory.")
