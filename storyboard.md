
# 🧠 Magic Commands

Thrive AI supports the following intelligent functions to assist with data exploration, analytics, and service navigation:

### 🔥 `correlation heatmap`
Generate a visual heatmap showing the correlation between variables in your dataset.

### ☁️ `word cloud`
Create a word cloud from text data to highlight frequently occurring terms.

### 📊 `descriptive statistics`
Compute basic summary statistics like mean, median, standard deviation, and more.

### 📈 `epidemiology`
Run population-level analytics for disease prevalence, incidence, or risk factor profiling.

### 📉 `principal component analysis`
Perform PCA to reduce dimensionality and identify dominant patterns in your data.

### 🧹 `feature reduction`
Apply statistical techniques to remove redundant or irrelevant features.

### 🗂️ `data catalogue`
Automatically generate metadata and schema documentation for your dataset.

### 📊 `visual dashboard`
Create interactive dashboards for data visualization using tools like Plotly or Streamlit.

### 🔄 `warehouse transformation`
Execute data transformation scripts on your enterprise data warehouse.

### 👥 `employee enrichment`
Integrate internal HR or staff data for population health workforce analysis.

### 🧠 `mind map`
Generate conceptual visual maps to represent topic relationships and themes.

### 🧬 `ontology`
Link your data to a healthcare or social determinants ontology for semantic enrichment.

### 🔍 `data science`
Run data science workflows like clustering, regression, or classification on selected data.

### 🚭 `smoking cessation`
Identify populations at risk and analyze outcomes related to tobacco cessation programs.

### 🧪 `diabetes risk analysis`
Evaluate individual or population-level risk for developing diabetes based on key indicators.

### 👨‍⚕️ `Find a provider PCP`
Locate a primary care provider based on geography, specialty, and insurance.

### 🧑‍⚕️ `Find a provider Specialist`
Identify a specialist provider based on patient needs and location.

### 🏘️ `Find a social needs`
Connect users with community-based organizations addressing social needs (e.g., food, housing, transportation).




## Magic architecture (MVC) 

1. **Identify Magic** - Automatically identifes major steps in notebook
2. **Load Data** - 1000 rows 95% accurate or  10000 rows 98.5% accurate or load whole table (100% accurate) 
3. **Perform Magic** - process Magic to produce nlp text or .png, or plot, or dataframe, or html 
4. **Render Magic Result** - render to produce .png, or plot, or html 
5. **Magic Controller** - special commands to work with magic results 





## AI assistent workflow

1. **Take in question** - import natural language  question
    Artifact 1 - Question
2. **Generate prompt for SQL** - Prompt engineers the NLP question into something more suitable for SQL generation 
    Artifact 2 - Prompt
3. **Generate SQL** - Using local LLM generate the SQL command 
    Artifact 3 - SQL command 
4. **Run SQL** - Run the SQL against the data warehouse 
    Artifact 4 - SQL results (in dataframe) 
5. **Convert results into pandas dataframe** - serialize the SQL results into a pandas dataframe 
    Artifact 5 - Persist Data Frame  
6. **Generate Matrix** - display the pandas dataframe somehow
    Artifact 6 - Persist Matrix  
7. **Generate Chart** - perform plotly through LLM 
    Artifact 7 - Persist Chart 
8. **Generate Summary** - perform summary through LLM 
    Artifact 8 - Persist Summary  
9. **Generate Insights** - perform insights through LLM 
    Artifact 9 - Persist Insights     
    





## AI assistent workflow

1. **Take in question** - import natural language  question
    - Artifact 1  **Question**
2. **Generate prompt for SQL** - Prompt engineers the NLP question into something more suitable for SQL generation 
    - Artifact 2  **Prompt**
3. **Generate SQL** - Using local LLM generate the SQL command 
    - Artifact 3  **SQL command**  
4. **Run SQL** - Run the SQL against the data warehouse 
    - Artifact 4  **SQL results** (in dataframe) 
5. **Convert results into pandas dataframe** - serialize the SQL results into a pandas dataframe 
    Artifact 5 - Persist Data Frame  
6. **Generate Matrix** - display the pandas dataframe somehow
    Artifact 6 - Persist Matrix  
7. **Generate Chart** - perform plotly through LLM 
    Artifact 7 - Persist Chart 
8. **Generate Summary** - perform summary through LLM 
    Artifact 8 - Persist Summary  
9. **Generate Insights** - perform insights through LLM 
    Artifact 9 - Persist Insights     
    





## AI assistent workflow

1. **Take in question** - import natural language  question
    - Artifact 1  **Question**
2. **Generate prompt for SQL** - Prompt engineers the NLP question into something more suitable for SQL generation 
    - Artifact 2  **Prompt**
3. **Generate SQL** - Using local LLM generate the SQL command 
    - Artifact 3  **SQL command**  
4. **Run SQL** - Run the SQL against the data warehouse 
    - Artifact 4  **SQL results** (in dataframe) 
5. **Convert results into pandas dataframe** - serialize the SQL results into a pandas dataframe 
    - Artifact 5  Persist **Data Frame**  
6. **Generate Matrix** - display the pandas dataframe somehow
    - Artifact 6 Persist **Matrix**  
7. **Generate Chart** - perform plotly through LLM 
    - Artifact 7  Persist Chart 
8. **Generate Summary** - perform summary through LLM 
    - Artifact 8 Persist Summary  
9. **Generate Insights** - perform insights through LLM 
    - Artifact 9 Persist Insights     
    





## AI assistent workflow

1. **Take in question** - import natural language  question
    - Artifact 1  **Question**
2. **Generate prompt for SQL** - Prompt engineers the NLP question into something more suitable for SQL generation 
    - Artifact 2  **Prompt**
3. **Generate SQL** - Using local LLM generate the SQL command 
    - Artifact 3  **SQL command**  
4. **Run SQL** - Run the SQL against the data warehouse 
    - Artifact 4  **SQL results** (in dataframe) 
5. **Convert results into pandas dataframe** - serialize the SQL results into a pandas dataframe 
    - Artifact 5  Persist **Data Frame**  
6. **Generate Matrix** - display the pandas dataframe somehow
    - Artifact 6 Persist **Matrix**  
7. **Generate Chart** - perform plotly through LLM 
    - Artifact 7  Persist **Chart** 
8. **Generate Summary** - perform summary through LLM 
    - Artifact 8 Persist **Summary**  
9. **Generate Insights** - perform insights through LLM 
    - Artifact 9 Persist **Insights**     
    





## AI assistent workflow - What is a research Canvas ??? 

1. **Take in question** - import natural language  question
    - Artifact 1  **Question**
2. **Generate prompt for SQL** - Prompt engineers the NLP question into something more suitable for SQL generation 
    - Artifact 2  **Prompt**
3. **Generate SQL** - Using local LLM generate the SQL command 
    - Artifact 3  **SQL command**  
4. **Run SQL** - Run the SQL against the data warehouse 
    - Artifact 4  **SQL results** (in dataframe) 
5. **Convert results into pandas dataframe** - serialize the SQL results into a pandas dataframe 
    - Artifact 5  Persist **Data Frame**  
6. **Generate Matrix** - display the pandas dataframe somehow
    - Artifact 6 Persist **Matrix**  
7. **Generate Chart** - perform plotly through LLM 
    - Artifact 7  Persist **Chart** 
8. **Generate Summary** - perform summary through LLM 
    - Artifact 8 Persist **Summary**  
9. **Generate Insights** - perform insights through LLM 
    - Artifact 9 Persist **Insights**     
    


