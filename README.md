# Resume Categorization using BERT

## Objective

This project aims to design, implement, and train a machine-learning model that automatically categorizes resumes based on their domain. The project also includes developing a script that processes a batch of resumes, categorizes them, and outputs results to both directory structures and a CSV file.

## Task Breakdown

### 1. Data Exploration and Preprocessing

- **Dataset**: The dataset used for this project is provided in the `Resume.csv` file.
- **Data Analysis**:
  - The distribution of resume categories was analyzed to understand the dataset's composition.
  - Text length (word count) was analyzed to identify any outliers or inconsistencies in the data.
  - A word cloud was generated to visualize the most common words in the resumes.

- **Preprocessing**:
  - Text data was cleaned and preprocessed using the `TextCleaner` class, which handles tasks such as:
    - Lowercasing text
    - Removing numbers, punctuation, HTML tags, URLs, and emojis
    - Fixing contractions and encoding issues
    - Removing excessive whitespace </br>
  Text Cleaning process is managed in the script [`scripts
/textcleaner.py`](https://github.com/AIFahim/ResumeCategorization/blob/master/scripts/textcleaner.py).
  - The dataset was split into training, validation, and test sets using stratified sampling to ensure each set is representative of the entire dataset. This process is managed in the script [`scripts/stratifiedtraintestsplit.py`](https://github.com/AIFahim/ResumeCategorization/blob/master/scripts/stratifiedtraintestsplit.py).

### 2. Model Selection and Training

#### Model: BERT (Bidirectional Encoder Representations from Transformers)

- **Rationale**: BERT is a state-of-the-art model in natural language processing tasks, including text classification. Its pre-trained nature allows for fine-tuning on specific tasks like resume categorization, making it an ideal choice for this project.
  
- **Training**:
  - The BERT model [`ahmedheakl/bert-resume-classification`](https://github.com/AIFahim/ResumeCategorization/blob/master/notebooks/training_notebook.ipynb) was fine-tuned for 11 epochs to achieve better results.
  - Fine-tuning was conducted using the training set, and the model's performance was validated on a separate validation set.
  - The best model, based on evaluation accuracy, was saved for inference.

#### Training Configuration:

- **Model ID**: `ahmedheakl/bert-resume-classification` from HuggingFace
- **Learning Rate**: 2e-5
- **Batch Size**: 8 (for both training and evaluation)
- **Epochs**: 11
- **Optimizer**: AdamW with weight decay
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### Model Checkpoint and Tokenizer

The fine-tuned BERT model checkpoint, along with the label encoder and tokenizer, has been saved and can be downloaded from Google Drive:

- [Model Checkpoint with Label Encoder and Tokenizer](https://drive.google.com/file/d/1oQD8tPI1svFcR9AjSW6ANRoXYaTLHDuJ/view?usp=sharing)

This includes everything needed for inference: the model weights, the tokenizer configuration, and the label encoder used during training.


### 3. Script Development

- **Script Name**: `script.py`
- **Functionality**:
  - The script accepts a directory of resumes (in PDF format), categorizes them using the trained BERT model, and moves them to their respective category folders (creating folders if necessary).
  - It also generates a CSV file named `categorized_resumes.csv` with two columns: `filename` and `category`.
- **Usage**:
python `scripts/script.py path/to/dir`


Replace `path/to/dir` with the path to the directory containing the PDF resumes.

### 4. Model and Data Handling

- **Model Directory**: The model is downloaded from Google Drive if not available locally, and extracted to a specified directory.
- **Inference**:
- The script loads the fine-tuned BERT model and uses it to categorize the resumes.
- The predictions are saved and the resumes are organized into corresponding category folders.

### 5. Evaluation Metrics

- **Evaluation**: The model's performance was evaluated using the test dataset.
- **Metrics**:
- **Accuracy**: Overall correctness of the model's predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positive predictions out of all actual positives.
- **F1-Score**: Harmonic mean of precision and recall, balancing the two.

### 6. Visualizations and Insights

### Visualizations and Insights

- **Category Distribution**: A bar chart visualizing the distribution of resume categories.

  ![Category Distribution](https://github.com/AIFahim/ResumeCategorization/blob/master/plots/category_distribution.png)

- **Word Count Distribution**: A histogram showing the distribution of word counts across resumes.

  ![Word Count Distribution](https://github.com/AIFahim/ResumeCategorization/blob/master/plots/word_count_distribution.png)

- **Word Cloud**: A word cloud visualization of the most common words in the resumes.

  ![Word Cloud](https://github.com/AIFahim/ResumeCategorization/blob/master/plots/word_cloud.png)


### 7. Deliverables

- **Jupyter Notebooks**: Provided detailing the exploration, preprocessing, and model training process.
- **Trained Model**: The fine-tuned BERT model saved for inference.
- **Script**: `script.py` for categorizing resumes.
- **Sample Output**: `categorized_resumes.csv` as a sample output after running the script on a test set.
- **Documentation**: This README file with instructions and details.

### Future Directions

- **Advanced Modeling**:
  - Consider using a larger language model (LLM) for even better classification performance.
  - Precompute text embeddings (using Sentence Transformer with Chunking) and use a machine learning or deep learning model for classification.

- **Further Fine-tuning**:
  - More epochs or hyperparameter tuning could improve the model's performance, especially in resource-constrained environments where fewer epochs were used.

- **Expanding Categories**:
  - The model can be expanded to handle more specific categories or industries as needed.


## How to Run

1. **Clone the Repository**:
git clone https://github.com/yourusername/resume-categorization.git cd resume-categorization


2. **Install Dependencies**:
pip install -r requirements.txt


3. **Run the Preprocessing and Training**:
Execute the notebooks or scripts provided to preprocess the data, train the model, and evaluate its performance.

4. **Run the Categorization Script**:
python script.py path/to/dir

5. **Review the Results**:
- Check the categorized resumes in the output directory.
- Review the `categorized_resumes.csv` file for the categorization results.

