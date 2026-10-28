
# 📰 Fake News Detection 

A machine-learning web application built with Python, Scikit-learn, and Streamlit to classify news headlines and articles as **Real-like** or **Fake-like** based on learned text patterns.

The project includes a Jupyter Notebook for model training and evaluation, a Streamlit web application for interactive predictions, and a live headline analysis feature using Google News RSS.

> **Disclaimer:** This application provides text-based machine-learning predictions. It does not independently verify facts or prove whether a news story is true or false.

---

## ✨ Features

- Text preprocessing and cleaning
- TF-IDF feature extraction
- Machine-learning news classification
- Comparison of Linear SVM, Logistic Regression, and Multinomial Naive Bayes
- Hyperparameter tuning using GridSearchCV
- Stratified cross-validation
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Interactive Streamlit dashboard
- Manual headline and article analysis
- Live headlines from Google News RSS
- Keyword-based headline search
- Original article links
- Downloadable news prediction results in CSV format
- Saved model pipeline using Joblib

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas and NumPy | Data processing |
| Scikit-learn | Machine-learning algorithms and evaluation |
| TF-IDF | Text feature extraction |
| Linear SVM | Text classification |
| Logistic Regression | Classification comparison |
| Multinomial Naive Bayes | Classification comparison |
| Streamlit | Web application |
| Matplotlib and Seaborn | Visualization |
| BeautifulSoup | RSS parsing and text cleaning |
| Requests | Retrieving live news |
| Joblib | Saving and loading the trained model |
| Git and GitHub | Version control and project hosting |

---

## 📂 Project Structure

```text
fake-news-detector/
│
├── fake_news_detector.ipynb
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── True.csv
├── Fake.csv
│
└── fake_news_model.joblib
```

### File descriptions

- `fake_news_detector.ipynb` — data preprocessing, model training, hyperparameter tuning, and evaluation.
- `app.py` — Streamlit application for news analysis.
- `requirements.txt` — required Python libraries.
- `True.csv` — dataset containing real-labelled news.
- `Fake.csv` — dataset containing fake-labelled news.
- `fake_news_model.joblib` — saved TF-IDF and classifier pipeline.
- `README.md` — project documentation.
- `LICENSE` — MIT License for the project code.
- `.gitignore` — files excluded from Git.

The CSV datasets and trained model may be excluded from GitHub if their licenses, file sizes, or distribution permissions require it. The application must still have access to a compatible model at runtime.

---

## 📋 Requirements

- Python 3.10 or a compatible version
- pip
- Git
- GitHub account
- Internet connection for live news retrieval

---

## 🚀 Installation and Setup

### Step 1: Clone the repository

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub username and repository name.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Step 2: Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

Create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
beautifulsoup4
requests
joblib
```

Install the libraries:

```bash
python -m pip install -r requirements.txt
```

For reproducible results, test your environment and pin compatible dependency versions before deployment.

---

## 🧠 Model Training

### Step 1: Prepare the datasets

Place `True.csv` and `Fake.csv` in the project directory.

Both CSV files must contain a `text` column with news article text.

### Step 2: Open the Jupyter Notebook

Install and start Jupyter:

```bash
python -m pip install notebook
jupyter notebook
```

Open `fake_news_detector.ipynb` and run the cells to:

1. Load and inspect the datasets.
2. Clean and preprocess the text.
3. Split the dataset into training and test sets.
4. Train and compare classification algorithms.
5. Perform cross-validation and hyperparameter tuning.
6. Evaluate the selected model.
7. Save the fitted model pipeline.

### Step 3: Save the trained model

If your notebook has already fitted variables named `tfidf` and `model`, save them together:

```python
import joblib
from sklearn.pipeline import Pipeline

news_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("classifier", model)
])

joblib.dump(
    news_pipeline,
    "fake_news_model.joblib"
)

print("Model saved successfully!")
```

Run this cell after fitting both objects.

If your notebook already trains and saves a complete scikit-learn pipeline, use that pipeline instead of creating a second one.

**Important:** The model's label mapping and text preprocessing must match the implementation in `app.py`.

The expected label mapping is:

- `1` = Real-like
- `0` = Fake-like

---

## ▶️ Run the Streamlit Application

After the model has been saved, execute:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

---

## 🖥️ Application Modules

### 1. Home

Displays the application overview, available features, and model status.

### 2. Check News

Paste a headline or news article into the text box and submit it for analysis.

Possible outputs:

- **Real-like:** the classifier predicts class `1`.
- **Fake-like:** the classifier predicts class `0`.
- **Insufficient text:** the submitted text is too short for the configured input check.

These labels describe model predictions, not verified factual conclusions.

### 3. Live News

Retrieves recent headlines from Google News RSS and applies the classifier to each headline.

The interface displays the headline, available publisher and publication information, model prediction, and original link when available.

### 4. About Model

Provides information about the classification pipeline, evaluation methods, and limitations.

---

## 📊 Model Evaluation

The notebook can report:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Classification report
- Confusion matrix
- Stratified cross-validation scores

A held-out test set should remain separate from model selection and hyperparameter tuning.

For a more realistic assessment, evaluate the classifier on newer articles that were not included in training. Random splits can overestimate performance when duplicate or closely related stories occur across the dataset.

No accuracy value is guaranteed; actual results depend on the dataset, preprocessing, selected algorithm, and test methodology.

---

## 🔍 Improving Prediction Reliability

The model learns statistical patterns from the training data. It does not automatically understand whether the claims in an article are factually correct.

Possible improvements include:

1. Remove duplicate and incorrectly labelled training examples.
2. Compare multiple algorithms using the same validation procedure.
3. Evaluate on newer, unseen news articles.
4. Analyze complete article text rather than relying only on headlines.
5. Check the original publisher, publication date, and independent reporting.
6. Use a review-needed outcome when evidence is insufficient.

A classifier's decision score should not be presented as a probability that an article is true unless it has been appropriately calibrated and validated for that interpretation.

---

## ☁️ Upload the Project to GitHub

### Method 1: Using the GitHub website

1. Sign in at [GitHub](https://github.com/).
2. Click **+ → New repository**.
3. Enter a repository name, such as `fake-news-detector`.
4. Add a short description.
5. Select Public or Private.
6. Create the repository.
7. Choose **Add file → Upload files**.
8. Upload `app.py`, `fake_news_detector.ipynb`, `requirements.txt`, `README.md`, `LICENSE`, and `.gitignore`.
9. Commit the changes.

Upload the datasets and model only if you have permission to redistribute them and their file sizes are appropriate.

### Method 2: Using Git commands

Initialize Git in your project directory:

```bash
git init
git branch -M main
```

Create a `.gitignore` file:

```gitignore
__pycache__/
*.py[cod]
venv/
.venv/
.ipynb_checkpoints/
.env
*.log

# Local datasets
True.csv
Fake.csv

# Model artifact
fake_news_model.joblib
```

The last three entries prevent the datasets and model from being committed. Remove those specific entries only if you intend and are permitted to publish those files.

Stage and commit your project:

```bash
git add app.py fake_news_detector.ipynb requirements.txt README.md LICENSE .gitignore
git commit -m "Add fake news detector project"
```

Connect your repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the remote already exists, inspect it with `git remote -v` rather than adding it again.

---

## 🌐 Deploy with Streamlit Community Cloud

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Create a new app.
5. Select your repository and branch.
6. Set the main file path to `app.py`.
7. Deploy the application.

Ensure the deployment environment has access to a compatible `fake_news_model.joblib`. If the model is excluded from the repository, configure a permitted way to supply it or retrain it during a separate preparation step.

---

## 📄 License

This project's original code is released under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the complete terms.

The license does not automatically grant rights to redistribute third-party datasets, pretrained artifacts, or external content. Check their respective licensing terms before publishing them.

---

## 👨‍💻 Author

Ninad Chinnawar

- GitHub: https://github.com/ninad-chinnawar

Replace the placeholders with your own details.

---

## ⭐ Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the application.
5. Submit a pull request describing the changes.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes. Predictions can be incorrect and should not be used as the sole basis for determining the credibility of a news report. Verify important claims using reliable, independent sources.
