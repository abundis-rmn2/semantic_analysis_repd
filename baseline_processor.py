# Import necessary libraries for data processing, machine learning, and text analysis
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import re  # Regular expressions
import nltk  # Natural Language Toolkit for text processing
from nltk.tokenize import word_tokenize  # Breaking text into words
from nltk.corpus import stopwords  # Common words to remove
from sklearn.model_selection import train_test_split  # Splitting data for training
from sklearn.feature_extraction.text import TfidfVectorizer  # Converting text to numerical features
from sklearn.ensemble import RandomForestClassifier  # Machine learning classifier
from sklearn.metrics import classification_report  # Performance evaluation

# Download necessary NLTK resources for text processing
nltk.download("stopwords")
nltk.download("punkt")


class ViolentKidnappingDetector:
    def __init__(self):
        # Define keywords that might indicate violence
        # Using prefixes to catch variations of words
        self.violent_keywords = {
            "arma",  # weapon
            "accident",  # accident
            "camioneta",  # van
            "llev",  # took/carrying (prefix for llevaron, llevado)
            "golp",  # hit/beat (prefix for golpeado)
            "secuestr",  # kidnapping
            "fuerz",  # force
            "rehabil",
            "amenaz",
            "arma",
            "dinero",
            "organ",
            "discu",
            "drog",
            "pistol",
            "empleo",
            "buscar trabajo",
            "carta",
            "violen",
            "camionera",
            "central"
        }

        # Create a comprehensive list of stopwords to remove
        # Combines Spanish stopwords with domain-specific words
        self.custom_stopwords = set(stopwords.words('spanish')).union({
            ",", "de", "la", "a", "su", "en", "que", "el", "se", "y", "del", ".",
            "las", "con", "domicilio", "reportante", "colonia", "paradero",
            "una", "lo", "ubicado", "municipio", "refiere", "jalisco",
            "domicilio", "persona", "jpg", "día"
        })

    def calculate_sum_score(self, row):
        """
        Calculates sum_score based on:
        - Adds 2 if 'condicion_localizacion' is "NO APLICA" or "SIN VIDA".
        - Counts violent terms using specific rules.
        """
        sum_score = 0

        # Count violent terms
        term_counts = {}
        for term in row['violence_terms']:
            base_term = next((kw for kw in self.violent_keywords if kw in term), term)
            term_counts[base_term] = term_counts.get(base_term, 0) + 1

        for count in term_counts.values():
            if count == 1:
                sum_score += 1
            else:
                sum_score += 0.5

        # Define score mapping for 'condicion_localizacion'
        condition_scores = {
            "CON VIDA": 0.5,
            "NO APLICA": 1,
            "SIN VIDA": 2
        }

        # Add the corresponding score if the condition exists in the dictionary
        sum_score += condition_scores.get(row['condicion_localizacion'], 0)

        return sum_score

    def preprocess_text(self, text):
        """
        Clean and tokenize text:
        1. Convert to lowercase
        2. Split into individual words
        3. Remove stopwords
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(str(text).lower())

        # Remove stopwords
        return [word for word in tokens if word not in self.custom_stopwords]

    def detect_violent_keywords(self, tokens):
        """
        Find words that match violent keyword prefixes
        Example:
        - "llevaron" matches "llev"
        - "golpeado" matches "golp"
        """
        return [word for word in tokens if
                any(keyword in word for keyword in self.violent_keywords)]

    def extract_features(self, df):
        """
        Prepare data for machine learning:
        1. Tokenize descriptions
        2. Detect violent terms
        3. Create violence label
        4. Convert text to numerical features (TF-IDF)
        """
        # Tokenize descriptions
        df['tokens'] = df['descripcion_desaparicion'].apply(self.preprocess_text)

        # Find violent terms using prefix matching
        df['violence_terms'] = df['tokens'].apply(self.detect_violent_keywords)

        # Label as violent if any violent terms found
        df['is_violent'] = df['violence_terms'].apply(lambda x: 1 if len(x) > 0 else 0)

        # Add violence score based on the number of violence terms
        df['violence_score'] = df['violence_terms'].apply(len)

        # Add a binary flag for whether there is at least one violent term
        df['has_violent_term'] = df['violence_terms'].apply(lambda x: 1 if len(x) > 0 else 0)

        # Convert text to numerical features using TF-IDF
        tfidf = TfidfVectorizer(
            stop_words=list(self.custom_stopwords),
            ngram_range=(1, 3),  # Consider 1-3 word combinations
            max_df=0.95,  # Ignore very common terms
            min_df=2  # Ignore very rare terms
        )
        tfidf_matrix = tfidf.fit_transform(df['descripcion_desaparicion'])

        return tfidf_matrix, tfidf

    def train_classifier(self, df, tfidf_matrix):
        """
        Train a machine learning model to classify violent cases:
        1. Split data into training and testing sets
        2. Train Random Forest classifier
        3. Evaluate model performance
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, df['is_violent'], test_size=0.2, random_state=42
        )

        # Create and train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions and print performance metrics
        y_pred = clf.predict(X_test)
        print("\n🔍 Classification Report:")
        print(classification_report(y_test, y_pred))

        return clf

    def process_data(self, df):
        """
        Main data processing pipeline:
        1. Extract features
        2. Train classifier
        3. Predict violent cases
        4. Save results with specific columns
        """
        # Extract text features
        tfidf_matrix, tfidf = self.extract_features(df)

        # Train machine learning classifier
        classifier = self.train_classifier(df, tfidf_matrix)

        # Predict violent cases
        df['predicted_violent'] = classifier.predict(tfidf_matrix)

        # Convert violence terms to a readable string
        df['violence_terms'] = df['violence_terms'].apply(lambda x: ', '.join(x))

        # Dentro de ViolentKidnappingDetector.process_data
        # Agregar columna sum_score
        df['sum_score'] = df.apply(lambda row: self.calculate_sum_score(row), axis=1)

        # Columns to export (only the specified columns)
        columns_to_export = [
            'id_cedula_busqueda',
            'condicion_localizacion',
            'edad_momento_desaparicion',
            'sexo',
            'genero',
            'descripcion_desaparicion',
            'violence_terms',
            'has_violent_term',  # Binary flag for at least one violent term
            'violence_score',
            'sum_score',
            'is_violent'
        ]

        # Check if all columns exist in the dataframe
        existing_columns = [col for col in columns_to_export if col in df.columns]

        # Export all cases (violent and non-violent)
        df_all = df[existing_columns]
        df_all.to_csv("filtered_cases_with_violence_terms.csv", index=False)

        return df_all


# Main execution
def main():
    # Load missing persons dataset
    df = pd.read_csv("./csv/sisovid.csv")

    # Create detector and process data
    detector = ViolentKidnappingDetector()
    filtered_cases = detector.process_data(df)

    # Display full details of detected cases
    pd.set_option('display.max_colwidth', None)
    print("\n🚨 Filtered Cases with Violence Terms:")
    print(filtered_cases)


if __name__ == "__main__":
    main()