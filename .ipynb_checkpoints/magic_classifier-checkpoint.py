import re
from rapidfuzz import fuzz
import yaml
from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Domain keywords organized for clarity and scalability

MAGIC_KEYWORDS = {
    "correlation heatmap": {
        # Visualize correlations between features
        "correlation", "heatmap", "correlation heatmap", "correlation matrix", "correlation coefficient", "show correlation"
    },
    "word cloud": {
        # Visualize word frequency
        "word cloud", "word map", "text cloud", "frequency cloud", "word visualization"
    },
    "descriptive statistics": {
        # Summarize numeric data
        "descriptive statistics", "describe data", "summary statistics", "statistically describe", "provide statistics", "basic stats"
    },
    "epidemiology": {
        # Public health patterns and analysis
        "epidemiology", "disease burden", "incidence", "prevalence", "population health", "epidemic trends"
    },
    "principal component analysis": {
        # Dimensionality reduction
        "principal component analysis", "component analysis", "PCA", "reduce dimensionality", "eigenvectors", "eigenvalues"
    },
    "feature reduction": {
        # Reduce number of features in dataset
        "feature reduction", "reduce features", "drop columns", "dimensionality reduction", "simplify features", "select features"
    },
    "data catalogue": {
        # Metadata and data inventory
        "data catalogue", "data catalog", "dataset list", "data dictionary", "field inventory", "metadata list"
    },
    "visual dashboard": {
        # Visual analytics
        "visual dashboard", "dashboard view", "interactive dashboard", "pygwalker", "data visualization", "plot dashboard"
    },
    "warehouse transformation": {
        # Data warehouse pipeline transformation
        "warehouse transformation", "data pipeline", "etl process", "transform warehouse", "data transformation"
    },
    "employee enrichment": {
        # Knowledge sharing and staff training
        "knowledge transfer", "employee enrichment", "staff education", "training", "learning module"
    },
    "mind map": {
        # Conceptual or brainstorming mapping
        "mind map", "concept map", "idea map", "thought map", "visual brainstorm"
    },
    "ontology": {
        # Structured relationships and taxonomies
        "ontology", "owlready2", "semantic network", "concept relationships", "taxonomies", "ontological model"
    },
    "data science": {
        # Structured relationships and taxonomies
        "principal component analysis", "feature reduction", "filtering", "group by" 
    }    
}

import yaml
import re
from pathlib import Path
from rapidfuzz import fuzz
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load and prepare keyword lists
def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Preprocess text (lowercase, stem)
stemmer = PorterStemmer()

def preprocess_text(text, use_stemming=False):
    text = text.lower()
    if use_stemming:
        tokens = word_tokenize(text)
        return ' '.join(stemmer.stem(token) for token in tokens)
    return text

# Core classification function
def classify_text(text, keywords_dict, use_fuzzy=False, use_stemming=False):
    text = preprocess_text(text, use_stemming)
    scores = {category: 0 for category in keywords_dict}
    matched_keywords = {category: [] for category in keywords_dict}

    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            # Preprocess keyword for stemming
            keyword_proc = preprocess_text(keyword, use_stemming)

            # Exact match
            if re.search(rf'\b{re.escape(keyword_proc)}\b', text):
                scores[category] += 1
                matched_keywords[category].append(keyword)
            # Optional: Fuzzy match
            elif use_fuzzy:
                similarity = fuzz.partial_ratio(keyword_proc, text)
                if similarity > 85:  # Adjustable threshold
                    scores[category] += 1
                    matched_keywords[category].append(f"{keyword} (fuzzy:{similarity}%)")

    total_matches = sum(scores.values())
    best_match = max(scores, key=scores.get)

    if scores[best_match] > 0:
        score = scores[best_match]
        confidence = round((score / total_matches) * 100, 2) if total_matches > 0 else 0.0
        evidence = ", ".join(matched_keywords[best_match])
    else:
        best_match = "general"
        score = 0
        confidence = 0.0
        evidence = ""

    return best_match, score, confidence, evidence

def magic_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, MAGIC_KEYWORDS, use_fuzzy, use_stemming )

def test_magic_classifier(sample_sentence):
    print(f"Text: {sample_sentence}")
    print(f"Domain: {domain_classifier(sample_sentence)}")
    print(f"Topic: {topic_classifier(sample_sentence)}\n")

      
 
def get_sample_questions():
    sentences = [
"What is the epidemology of breast cancer",
"Risk stratify the patients most likely to hospitalize in 30 days",
"Provide the list of tables related to patients",
"Provide a mind map regarding the etiology of cancer",
"Provide descriptive statistics on the provider table",
"Provide a dashboard about smokers ",
"provide data scientifically engineered for knowledge expansion ",
"Can the platform make recommendations on how to make warehouse more efficient ",

    ]
    return sentences

def run_magic_sample_test():
    sample_sentences = get_sample_questions()
    for sentence in sample_sentences:
        test_magic_classifier(sentence)
    return    

