🎭 AI-Powered Emotion Detection from Text



📖 Overview



Emotion is one of the basic instincts of human beings. This project aims to detect and recognize feelings from textual data using Machine Learning. By analyzing the expressions in sentences, the model classifies text into specific emotional states such as anger, surprise, joy, fear, sadness, and love.



This solution utilizes Natural Language Processing (NLP) techniques for text preprocessing and Logistic Regression for classification.



📂 Dataset



The project uses a dataset containing 20,000 entries of text and their corresponding sentiment labels.



Input: Textual content (e.g., tweets, comments).



Target: Sentiment class (6 categories).



Classes: anger, fear, joy, love, sadness, surprise.



🛠️ Tech Stack



* Language: Python



* Environment: Jupyter Notebook / Google Colab



* Libraries:



&nbsp;        Pandas \& NumPy (Data Manipulation)



&nbsp;        Seaborn \& Matplotlib (Visualization)



&nbsp;        NLTK (Text Preprocessing: Stopwords, Lemmatization)



&nbsp;        Scikit-learn (Model Building \& Evaluation)



⚙️ Methodology



The project follows a structured data science pipeline:



1. Data Collection \& Cleaning: Loading the dataset and handling missing values.



2. Text Preprocessing:

&nbsp;

* 🧹 Punctuation Removal: Stripping special characters and numbers.



* 😀 Emoji Handling: Replacing emojis with their text descriptions.



* ✂️ Tokenization: Splitting sentences into words.



* 🚫 Stopword Removal: Filtering out common English words.



* 🌱 Lemmatization: Converting words to their base root form.



3.  Feature Extraction: Converting text data into numerical vectors using TF-IDF (Term           Frequency-Inverse Document Frequency).



4.   Model Training: Training a Logistic Regression classifier on the processed data.





📊 Results



The model achieves high performance in distinguishing between different emotions.



Metric	        score

Accuracy	86.27%

Precision	88.31%

Recall	        76.18%

F1-Score	80.54%



Note: Detailed classification reports and confusion matrices are available in the notebook.



🚀 How to Run



1. Clone the repository:



**git clone https://github.com/your-username/emotion-detection.git**



2\.  Install dependencies:



pip install pandas numpy seaborn matplotlib nltk scikit-learn



3\.  Open the Notebook: Launch

&nbsp;

AI\_Powered\_Emotion\_Detection\_from\_Text.ipynb in Jupyter Notebook or Google Colab.



4\.  Run the cells: Execute the cells sequentially to preprocess data, train the model, and see results.












