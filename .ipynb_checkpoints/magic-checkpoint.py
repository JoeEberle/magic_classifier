import re
from rapidfuzz import fuzz
import yaml
from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import postgres_utils as pg
import matplotlib.pyplot as plt   # matplot is for python graphics
import numpy as np   #numpy is for array processing
import seaborn  as sns
import PIL.Image
from wordcloud import WordCloud, STOPWORDS
import domain_topic_classifier as dtc
stopwords = STOPWORDS

# Domain keywords organized for clarity and scalability

MAGIC_KEYWORDS = {
    "correlation heatmap": {
        # Visualize correlations between features
        "correlation", "heatmap", "correlation heatmap", "correlation matrix", "correlation coefficient", "show correlation","\correlation heatmap"
    },
    "word cloud": {
        # Visualize word frequency
        "word cloud", "word map", "text cloud", "frequency cloud", "word visualization","\word cloud"
    },
    "descriptive statistics": {
        # Summarize numeric data
        "descriptive statistics", "describe data", "summary statistics", "statistically describe", "provide statistics", "basic stats"
        , "\descriptive statistics"
    },
    "epidemiology": {
        # Public health patterns and analysis
        "epidemiology", "disease burden", "incidence", "prevalence", "population health", "epidemic trends","\epidemiology"
    },
    "principal component analysis": {
        # Dimensionality reduction
        "principal component analysis", "component analysis", "PCA", "reduce dimensionality", "eigenvectors", "eigenvalues"
    },
    "feature reduction": {
        # Reduce number of features in dataset
        "feature reduction", "reduce features", "drop columns", "dimensionality reduction", "simplify features", "select features"
    },
    "data catalog": {
        # Metadata and data inventory
        "data catalogue", "data catalog", "dataset list", "data dictionary", "field inventory", "metadata list", "table", "columns", "schema"
    },
    "visual dashboard": {
        # Visual analytics
        "visual dashboard", "dashboard view", "interactive dashboard", "pygwalker", "data visualization", "plot dashboard",  "dashboard"
    },
    "warehouse transformation": {
        # Data warehouse pipeline transformation
        "warehouse transformation", "data pipeline", "etl process", "transform warehouse", "data transformation", "warehouse", "transformation"
    },
    "employee enrichment": {
        # Knowledge sharing and staff training
        "knowledge transfer", "employee enrichment", "staff education", "training", "learning module"
    },
    "mind map": {
        # Conceptual or brainstorming mapping
        "/mind map", "concept map", "idea map", "thought map", "visual brainstorm"
    },
    "histogram": {
        # Conceptual or brainstorming mapping
        "histogram", "/histogram", "/histo", 
    },
    "ontology": {
        # Structured relationships and taxonomies
        "ontology", "owlready2", "semantic network", "concept relationships", "taxonomies", "ontological model"
    },
    "data science": {
        # Structured relationships and taxonomies
        "principal component analysis", "feature reduction", "filtering", "group by" 
    }, 
    "pair plot": {
        # Structured relationships and taxonomies
        "pair plot", "/pair plot", "\pair plot", "pair visualize" 
    },     
    "risk stratify": {
        # Structured relationships and taxonomies
        "risk stratify", "risk", "stratify" 
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

def find_magic_table_name(domain):
    table_name = "wny_health"
    if domain == 'healthcare':
        table_name = "wny_health"
    elif domain == 'penguin':  
        table_name = "penguin"   
    elif domain == 'titanic':  
        table_name = "titanic" 
    else:
        table_name = "wny_health"      
    return table_name  

def magic_data_load(table_name):
    DB_NAME, USER, PASSWORD, HOST, PORT = pg.get_connection_credentials()
    connection = pg.connect_to_postgresql(DB_NAME, USER, PASSWORD, HOST, PORT)     
    load_sql = f"select * from {table_name} limit 10000"
    df_magic = pg.query_to_dataframe(load_sql , connection)
    return df_magic


def perform_magic_correlation_heatmap(df):
    correlation_matrix = df.corr(numeric_only=True)     #establish a correlation matrix for all fields
    top_correlation_features = correlation_matrix.index
    plt.figure(figsize=(8,8))
    g=sns.heatmap(df[top_correlation_features].corr(),annot=True,cmap="RdYlGn")
    insights = heatmap_insights_generator(df) 
    return g, insights


def perform_magic_histogram(df):
    g = df.hist(figsize=(10,8), bins=20)
    return g

def perform_magic_pairplot(df, hue_value):
    g = sns.pairplot(data=df, hue=hue_value)
    return g

def perform_magic_word_cloud(df, shape_image_file):
    
    shape = np.array(PIL.Image.open(shape_image_file))
    text_columns = df.select_dtypes(include=['object', 'string']).columns[:3]       # Get the first 3 text columns
    cloud_text = ' '.join(df[col].dropna().astype(str).str.cat(sep=' ') for col in text_columns)

    wc = WordCloud(
            background_color ='white',
            stopwords=stopwords,
            mask = shape,
            height=600,
            width=400,
            contour_color='green',
            contour_width=10
    )
    
    wc.generate(cloud_text)
    g = plt.imshow(wc)
    return g


def perform_magic(magic_command, table_name):
    df_magic = magic_data_load(table_name)
    if magic_command == 'correlation heatmap':
        perform_magic_correlation_heatmap(df_magic)
    if magic_command == 'histogram':
        perform_magic_histogram(df_magic)        
    if magic_command == 'pair plot':
        if table_name == 'penguin':
            perform_magic_pairplot(df_magic, 'species')
        if table_name == 'titanic':
            perform_magic_pairplot(df_magic, 'sex')   
        if table_name == 'wny_health':
            perform_magic_pairplot(df_magic, 'sex')    
    if magic_command == 'word cloud':
        if table_name == 'penguin':
            perform_magic_word_cloud(df_magic, 'heart.png')
        if table_name == 'titanic':
            perform_magic_word_cloud(df_magic, 'heart.png')   
        if table_name == 'wny_health':
            perform_magic_word_cloud(df_magic, 'heart.png')     

def run_magic(sentence):
    domain_class,domain_score,domain_confidence,domain_evidence = dtc.domain_classifier(sentence, True, True)
    magic_class,magic_score,magic_confidence,magic_evidence = magic_classifier(sentence, True, True)
    magic_table = find_magic_table_name(domain_class)  
    graph = perform_magic(magic_class, magic_table)
        

def heatmap_insights_generator(df, threshold=0.7, max_pairs=10):
    """
    Generate Markdown summary of strongest correlations in a dataframe.
    
    Parameters:
    df (DataFrame): Input pandas dataframe
    threshold (float): Correlation threshold to report
    max_pairs (int): Maximum number of pairs to report
    
    Returns:
    str: Markdown formatted summary
    """
    corr = df.corr(numeric_only=True)
    corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
    
    # Remove self-correlations
    corr_filtered = corr_unstacked[corr_unstacked < 1.0]
    
    # Filter strong correlations above threshold
    strong_corr = corr_filtered[corr_filtered >= threshold].drop_duplicates().head(max_pairs)
    
    if strong_corr.empty:
        markdown = "### 🔍 Correlation Insights\n\nNo strong correlations (>|{}|) found in the dataset.".format(threshold)
        return markdown

    markdown = "### 🔍 Correlation Insights\n\n"
    markdown += "| Feature 1 | Feature 2 | Correlation |\n"
    markdown += "|-----------|-----------|-------------|\n"
    
    for (var1, var2), value in strong_corr.items():
        markdown += f"| `{var1}` | `{var2}` | **{value:.2f}** |\n"
    
    markdown += "\n**Summary:**\n"
    markdown += f"- {len(strong_corr)} strong relationships identified above threshold of {threshold}.\n"
    markdown += "- These variables may reflect redundancy, causality, or strong associations valuable for modeling or clinical interpretation.\n"
    markdown += "- Consider these pairs for feature selection or deeper domain review."
    
    return markdown

# Example usage:
# print(heatmap_insights_generator(df_penguins))

# Pseudo-code overview

# # 1. LangChain Tool Definitions
# @tool
# def generate_correlation_heatmap(df): ...

# @tool
# def generate_histogram(df): ...

# @tool
# def generate_wordcloud(df): ...

# @tool
# def load_magic_data(table_name): ...

# # 2. Router chain (uses prompt to pick tool)
# router_prompt = PromptTemplate.from_template("Choose the best tool for: {magic_command}")
# router_chain = LLMRouterChain(llm, tools=[...], prompt=router_prompt)

# # 3. LangGraph Nodes
# nodes = {
#     "load_data": lambda state: load_magic_data(state['table_name']),
#     "select_tool": router_chain,
#     "execute_tool": lambda state: state['tool'](state['df']),
# }

# graph = LangGraph(nodes, start="load_data", end="execute_tool")





