import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# Imports for PCA and KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Additional imports
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns


import plotly.express as px

import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Function to preprocess data
@st.cache_data
def preprocess_data(selected_features=None):
    df = pd.read_csv('heart_disease_uci.csv')

    # Preprocessing steps as before
    df.columns = ['age', 'is_male', 'chest_pain', 'rest_bp', 'chol', 'high_sugar', 'rest_ecg', 'max_hr', 'exercise_angina', 'st_depression', 'st_slope', 'num_fluoro', 'thalass_type', 'art_blocks']
    df = df.drop('num_fluoro', axis=1)
    df.at[87, 'thalass_type'] = 3.0
    df.at[266, 'thalass_type'] = 7.0
    df['is_heart_disease'] = (df['art_blocks'] > 0).astype(int)
    df = df.drop('art_blocks', axis=1)

    # Define feature types
    numeric_features = ['age', 'rest_bp', 'chol', 'max_hr', 'st_depression']
    categorical_features = ['chest_pain', 'rest_ecg', 'st_slope', 'thalass_type']
    binary_features = ['is_male', 'high_sugar', 'exercise_angina']
    target_feature = 'is_heart_disease'

    all_features = numeric_features + categorical_features + binary_features
    if selected_features is not None:
        features_to_use = [feature for feature in all_features if feature in selected_features]
    else:
        features_to_use = all_features
        
    # Setting up preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])

    # Preprocess the data
    X = df[numeric_features + categorical_features + binary_features]
    X = preprocessor.fit_transform(X)
    y = df[target_feature]

    return X, y, preprocessor

from sklearn.metrics import silhouette_score

@st.cache_data
def calculate_elbow_curve(X, K_max):
    inertia = []
    silhouette_scores = []
    for k in range(2, K_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    return inertia, silhouette_scores

@st.cache_data
def compute_model_accuracies(X, y, _classifiers, num_iterations, selected_features=None):
    if selected_features is None:
        num_features = X.shape[1]
        feature_indices = np.arange(num_features)
    else:
        num_features = len(selected_features)
        # Map selected feature names to their indices in the full feature set
        all_features = ['age', 'rest_bp', 'chol', 'max_hr', 'st_depression', 'chest_pain', 'rest_ecg', 'st_slope', 'thalass_type', 'is_male', 'high_sugar', 'exercise_angina']
        feature_indices = [all_features.index(feature) for feature in selected_features]

    classifier_scores = {name: np.zeros(num_features) for name in _classifiers}

    for name, model in _classifiers.items():
        for iteration in range(num_iterations):
            np.random.shuffle(feature_indices)
            for i in range(1, num_features + 1):
                selected_feature_indices = feature_indices[:i]
                X_train, X_test, y_train, y_test = train_test_split(X[:, selected_feature_indices], y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                classifier_scores[name][i-1] += accuracy
        classifier_scores[name] /= num_iterations
    return classifier_scores

    

# Function to train and evaluate a model with given hyperparameters
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, report, confusion


# Functions for PCA, KMeans, plotting clusters, and adding cluster labels
@st.cache_data
def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

@st.cache_data
def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X)

def plot_pca_clusters(X_pca, labels):
    df = pd.DataFrame({'PCA Component 1': X_pca[:, 0], 'PCA Component 2': X_pca[:, 1], 'Labels': labels})
    fig = px.scatter(df, x='PCA Component 1', y='PCA Component 2', color='Labels', color_continuous_scale='viridis')
    fig.update_layout(title='PCA of Dataset with K-means Clusters',
                      xaxis_title='PCA Component 1',
                      yaxis_title='PCA Component 2')
    st.plotly_chart(fig)

def add_cluster_labels(X, labels):
    return np.hstack([X, labels.reshape(-1, 1)])

def evaluate_models_on_data(X, y, classifiers):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_scores = {}
    for name, model in classifiers.items():
        accuracy, _, _ = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        model_scores[name] = accuracy
    return model_scores

def plot_confusion_matrix_heatmap(confusion_matrix, title):
    fig = px.imshow(confusion_matrix,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    text_auto=True)
    fig.update_layout(title=title, coloraxis_showscale=False)
    st.plotly_chart(fig)

def plot_roc_curve(models, X_test, y_test):
    fig = go.Figure()
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc_score:.2f})'))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig)

def plot_precision_recall_curve(models, X_test, y_test):
    fig = go.Figure()
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        ap_score = average_precision_score(y_test, y_pred_prob)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'{name} (AP = {ap_score:.2f})'))

    fig.update_layout(title='Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision')
    st.plotly_chart(fig)

def plot_feature_importance(importances, feature_names, title):
    fig = px.bar(x=importances, y=feature_names, labels={'x': 'Importance', 'y': 'Feature'}, orientation='h')
    fig.update_layout(title=title)
    st.plotly_chart(fig)

    
# Setup
st.title('Heart Disease Prediction with K-Means Clustering')

# Data preprocessing
X, y, _ = preprocess_data()
# Caching the elbow curve calculation

# Create Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Model Introduction", "Model Comparison", "Model Tuning", "Model Evaluation"])
# Tab 0: Introduction and Context Setting
with tab0:
    st.header("Understanding Heart Disease and the Role of Machine Learning")
    
    st.write("""
    ### The Impact of Heart Disease
    Heart disease is one of the leading causes of death globally. It encompasses a range 
    of conditions affecting the heart, including coronary artery disease, arrhythmias, 
    heart failure, and others. Early detection and prevention are key to reducing the 
    mortality rate associated with heart disease.
    
    ### How Machine Learning Can Help
    Machine learning (ML) offers promising tools in the fight against heart disease. By 
    analyzing vast amounts of health data, ML algorithms can identify patterns and 
    indicators that are often too complex or subtle for traditional analysis. These insights 
    can lead to early prediction and diagnosis, personalized treatment plans, and better 
    patient outcomes.

    ### Our Approach
    In this application, we explore how different machine learning models can be used 
    to predict the risk of heart disease. We delve into various aspects of ML, from 
    data preprocessing and feature selection to model tuning and evaluation. Our goal 
    is to demonstrate the potential of ML in healthcare and provide an interactive 
    experience for understanding its impact.
    
    ### Navigating the Application
    - **Tab 1:** Explore data preprocessing and initial model evaluation.
    - **Tab 2:** Dive into model tuning with adjustable hyperparameters.
    - **Tab 3:** Investigate how k-means clustering can enhance model accuracy.
    - **Tab 4:** View additional insights and conclusions.

    Let's embark on this journey to uncover the potential of machine learning in 
    predicting heart disease and saving lives.
    """)

    with st.expander("About Kevin Mok"):
        st.write("""
            Kevin Mok, a microbiologist with a Bachelor of Science from Michigan State University, 
            has a rich background in scientific research and technology. His career spans roles 
            as a Biotechnician at MSU's Bioeconomy Institute and as an Infantryman in the U.S. Army, 
            where he honed skills in data analysis, biosafety protocols, and effective communication.
    
            In the realm of microbiology, Kevin's work has centered on fermentation labs and process 
            development teams, equipping him with hands-on experience in various microbial techniques. 
            He has adeptly used programming for data processing, showcasing his ability to blend 
            scientific expertise with technological solutions.
        """)

    with st.expander("Vision in Data Science and Bioinformatics")
        st.write("""
            Currently pursuing a Master's degree in Data Science at Michigan State University, Kevin 
            is poised to be at the forefront of the emerging field of bioinformatics. He envisions data 
            science as the fourth pillar of scientific discovery, particularly in unraveling the complexities 
            of genetic sequencing and the microbiome.
    
            With proficiency in Python, R, SQL, and a growing expertise in bioinformatics, Kevin aspires 
            to leverage data science in advancing our understanding of biology. His goal is to contribute 
            significantly to healthcare and environmental studies, utilizing data to drive innovative 
            research and discovery in these critical areas.
        """)


with tab1:
    st.header("Exploring Machine Learning for Heart Disease Analysis")
    st.write("""
        Welcome to our exploration of machine learning in the context of heart disease prediction. 
        This section provides an overview of how different features influence the accuracy of 
        machine learning models and highlights key features that are crucial in predicting heart disease.
    """)
  
    st.header("K-Means Clustering and Model Accuracy by Number of Features")

   # K-Means Elbow Curve and Silhouette Scores
    st.subheader("Finding the Optimal Number of Clusters")
    st.write("""
        Clustering helps in understanding data patterns, and choosing the right number of clusters 
        is important for meaningful analysis. We use two approaches to find the best number:
        
        - **Elbow Method**: Helps identify a point where adding more clusters brings little 
          improvement in modeling the data.
        - **Silhouette Score**: Measures how well data is clustered, with higher scores 
          indicating clearer, better-defined clusters.
        
        These methods assist in selecting a suitable number of clusters for our data analysis.
    """)
    K_max = 10
    inertia, silhouette_scores = calculate_elbow_curve(X, K_max)
    elbow_df = pd.DataFrame({
        'Number of Clusters': range(2, K_max + 1),
        'Inertia': inertia,
        'Silhouette Score': silhouette_scores
    })

    # Creating subplots using Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for Inertia and Silhouette Score
    fig.add_trace(
        go.Scatter(x=elbow_df['Number of Clusters'], y=elbow_df['Inertia'], name="Inertia"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=elbow_df['Number of Clusters'], y=elbow_df['Silhouette Score'], name="Silhouette Score"),
        secondary_y=True,
    )

    # Add figure title and axis labels
    fig.update_layout(
        title_text="K-Means Elbow Curve and Silhouette Scores"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Number of Clusters")

    # Set y-axes titles
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    # Classifier Accuracy by Number of Features
    st.subheader("Assessing the Impact of Feature Quantity on Model Accuracy")
    st.write("""
        In machine learning, the number of features used can significantly impact model accuracy. 
        This part of the analysis looks into:
        
        - How varying the number of features affects the accuracy of the models.
        - Observing the point at which adding more features yields diminishing returns in accuracy.
        
        This helps in determining an effective number of features for the models.
    """)

    classifiers = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier()
    }
    num_iterations = 10
    classifier_scores = compute_model_accuracies(X, y, classifiers, num_iterations)
    accuracy_df = pd.DataFrame({name: scores for name, scores in classifier_scores.items()})
    accuracy_df['Number of Features'] = range(1, X.shape[1] + 1)
    fig2 = px.line(accuracy_df, x='Number of Features', y=accuracy_df.columns[:-1], markers=True)
    fig2.update_layout(title='Classifier Accuracy vs Number of Features', xaxis_title='Number of Features', yaxis_title='Accuracy')
    st.plotly_chart(fig2)
    
    # Preprocess the full dataset
    X, y, preprocessor = preprocess_data()  # Assuming this function returns the full dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train tree-based models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    feature_names = preprocessor.get_feature_names_out()  # Update with your preprocessor
    st.subheader("Evaluating Feature Significance")
    st.write("""
        Understanding which features most significantly predict heart disease is crucial in model development:
        
        - **Decision Trees** highlight the most important features in making predictions.
        - **Random Forests** provide a more comprehensive view by aggregating insights from multiple trees.
        
        Analyzing feature importance is key to enhancing model performance and gaining insights into heart disease.
    """)
    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, 'feature_importances_'):
            st.subheader(f'{name} Feature Importances')
            plot_feature_importance(model.feature_importances_, feature_names, f'{name} Feature Importances')
        else:
            st.write(f'Feature importances are not available for the {name} model.')



# Model Comparison Tab
with tab2:
    st.header("Exploring the Influence of Feature Quantity on Model Accuracy")
    st.write("""
        In this section, we examine the relationship between the number of features used in our models and their predictive accuracy. 

        - **Incremental Feature Addition**: By progressively increasing the number of features in our models, 
          we can observe how each addition contributes to overall performance. This helps in understanding 
          the collective impact of features on the model's accuracy.

        - **Identifying Optimal Feature Set**: This analysis also helps in identifying the optimal number 
          of features where the model achieves high accuracy before the benefits of adding more features 
          begin to diminish.

        This approach offers insights into how model performance is affected by the quantity of features, 
        allowing us to strike a balance between model complexity and predictive power.
    """)
    st.header("Feature Selection and Model Analysis")

    # Feature Selection
    st.subheader("Select Features")
    all_possible_features = ['age', 'rest_bp', 'chol', 'max_hr', 'st_depression', 'chest_pain', 'rest_ecg', 'st_slope', 'thalass_type', 'is_male', 'high_sugar', 'exercise_angina']
    selected_features = st.multiselect("Choose features to include:", all_possible_features, default=all_possible_features)

    # Data Preprocessing with selected features
    X_reduced, y_reduced, _ = preprocess_data(selected_features)


    # Classifier Accuracy by Number of Features
    st.subheader("Classifier Accuracy by Number of Features")
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier()
    }
    num_iterations = 10
    classifier_scores = compute_model_accuracies(X_reduced, y_reduced, classifiers, num_iterations, selected_features)
    accuracy_df = pd.DataFrame({name: scores for name, scores in classifier_scores.items()})
    accuracy_df['Number of Features'] = range(1, len(selected_features) + 1 if selected_features else X_reduced.shape[1] + 1)
    fig2 = px.line(accuracy_df, x='Number of Features', y=accuracy_df.columns[:-1], markers=True)
    fig2.update_layout(title='Classifier Accuracy vs Number of Features', xaxis_title='Number of Features', yaxis_title='Accuracy')
    st.plotly_chart(fig2)
    
    
    

# Assuming the rest of your necessary imports and function definitions are here

with tab3:
    st.header("Model Tuning")
    st.header("Model Tuning with Hyperparameters")
    st.write("""
    Here, you can adjust the hyperparameters of various models to see how they 
    influence the model's performance. Use the sliders to change the parameters 
    and retrain the models to compare their accuracies.
    """)

    # Preselected features
    preselected_features = ['max_hr', 'age', 'st_depression', 'thalass_type', 'chest_pain', 'rest_bp']

    # Data preprocessing for all features and preselected features
    X_all, y_all, _ = preprocess_data()
    X_preselected, y_preselected, _ = preprocess_data(preselected_features)

    # Splitting the datasets
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train_preselected, X_test_preselected, y_train_preselected, y_test_preselected = train_test_split(X_preselected, y_preselected, test_size=0.2, random_state=42)

    # Hyperparameters input for each model
    with st.expander("KNN Hyperparameters"):
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key='knn_n_neighbors')
    
    with st.expander("Decision Tree Hyperparameters"):
        dt_max_depth = st.slider("Max Depth", 1, 20, 5, key='dt_max_depth')
        dt_min_samples_split = st.slider("Min Samples Split", 2, 10, 2, key='dt_min_samples_split')
    
    with st.expander("SVM Hyperparameters"):
        svm_C = st.slider("C (Regularization)", 0.01, 1.0, 0.1, key='svm_C')
        svm_kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key='svm_kernel')
    
    with st.expander("Random Forest Hyperparameters"):
        rf_n_estimators = st.slider("Number of Trees", 10, 100, 10, key='rf_n_estimators')
        rf_max_depth = st.slider("Max Depth", 1, 20, 5, key='rf_max_depth')

    # Train and Evaluate All Models Button
    if st.button("Train and Evaluate All Models"):
        # Setting up models with updated hyperparameters
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
            "Decision Tree": DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split),
            "SVM": SVC(C=svm_C, kernel=svm_kernel, probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
        }

        # Training and evaluating each model on both datasets
        model_performances_all = {}
        model_performances_preselected = {}
        roc_data_all = {}
        roc_data_preselected = {}
        accuracy_data_all = {}
        accuracy_data_preselected = {}
        pr_data_all = {}  # Precision-recall data for all features
        pr_data_preselected = {}  # Precision-recall data for preselected features

        for name, model in models.items():
            # Training and evaluating on all features
            model.fit(X_train_all, y_train_all)
            y_pred_all = model.predict(X_test_all)
            accuracy_all = accuracy_score(y_test_all, y_pred_all)
            model_performances_all[name] = {"accuracy": accuracy_all}
            # Collecting ROC data for all features
            y_pred_prob_all = model.predict_proba(X_test_all)[:, 1]
            fpr_all, tpr_all, _ = roc_curve(y_test_all, y_pred_prob_all)
            auc_score_all = auc(fpr_all, tpr_all)
            roc_data_all[name] = (fpr_all, tpr_all, auc_score_all)

            # Training and evaluating on preselected features
            model.fit(X_train_preselected, y_train_preselected)
            y_pred_preselected = model.predict(X_test_preselected)
            accuracy_preselected = accuracy_score(y_test_preselected, y_pred_preselected)
            model_performances_preselected[name] = {"accuracy": accuracy_preselected}
            # Collecting ROC data for preselected features
            y_pred_prob_preselected = model.predict_proba(X_test_preselected)[:, 1]
            fpr_preselected, tpr_preselected, _ = roc_curve(y_test_preselected, y_pred_prob_preselected)
            auc_score_preselected = auc(fpr_preselected, tpr_preselected)
            roc_data_preselected[name] = (fpr_preselected, tpr_preselected, auc_score_preselected)

        for name, model in models.items():
            # ROC data for all features
            y_pred_prob_all = model.predict_proba(X_test_all)[:, 1]
            fpr_all, tpr_all, _ = roc_curve(y_test_all, y_pred_prob_all)
            auc_score_all = auc(fpr_all, tpr_all)
            roc_data_all[name] = (fpr_all, tpr_all, auc_score_all)

            # ROC data for preselected features
            y_pred_prob_preselected = model.predict_proba(X_test_preselected)[:, 1]
            fpr_preselected, tpr_preselected, _ = roc_curve(y_test_preselected, y_pred_prob_preselected)
            auc_score_preselected = auc(fpr_preselected, tpr_preselected)
            roc_data_preselected[name] = (fpr_preselected, tpr_preselected, auc_score_preselected)
            
        accuracy_data_all = {}
        accuracy_data_preselected = {}
        pr_data_all = {}  # Precision-recall data for all features
        pr_data_preselected = {}  # Precision-recall data for preselected features

        for name, model in models.items():
            # Collecting accuracy data
            accuracy_data_all[name] = model_performances_all[name]["accuracy"]
            accuracy_data_preselected[name] = model_performances_preselected[name]["accuracy"]

            # Collecting precision-recall data
            precision_all, recall_all, _ = precision_recall_curve(y_test_all, model.predict_proba(X_test_all)[:, 1])
            precision_preselected, recall_preselected, _ = precision_recall_curve(y_test_preselected, model.predict_proba(X_test_preselected)[:, 1])
            pr_data_all[name] = (precision_all, recall_all)
            pr_data_preselected[name] = (precision_preselected, recall_preselected)

        # Plotting accuracy for all features
        fig_accuracy_all = go.Figure(data=[
            go.Bar(name=model_name, x=["Accuracy"], y=[accuracy_data_all[model_name]])
            for model_name in accuracy_data_all
        ])
        fig_accuracy_all.update_layout(title="Accuracy (All Features)")
        st.plotly_chart(fig_accuracy_all)

        # Plotting accuracy for preselected features
        fig_accuracy_preselected = go.Figure(data=[
            go.Bar(name=model_name, x=["Accuracy"], y=[accuracy_data_preselected[model_name]])
            for model_name in accuracy_data_preselected
        ])
        fig_accuracy_preselected.update_layout(title="Accuracy (Preselected Features)")
        st.plotly_chart(fig_accuracy_preselected)

        st.subheader("Precision-Recall Curves Analysis")
        st.write("""
            Precision-Recall curves are another important metric for evaluating the performance of classification models, especially in imbalanced datasets.
    
            - **Precision and Recall**: Precision measures the proportion of positive identifications that were 
              actually correct, while recall measures the proportion of actual positives that were identified correctly.
              
            - **Curve Interpretation**: A high area under the curve represents both high recall and high precision. 
              High precision relates to a low false positive rate, and high recall relates to a low false negative rate. 
    
            These curves provide insight into the trade-off between precision and recall for different threshold values, 
            helping in the selection of an optimal threshold for decision making.
        """)
        # Plotting precision-recall for all features
        fig_pr_all = go.Figure()
        for name, (precision, recall) in pr_data_all.items():
            fig_pr_all.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name))
        fig_pr_all.update_layout(title='Precision-Recall Curves (All Features)')
        st.plotly_chart(fig_pr_all)

        # Plotting precision-recall for preselected features
        fig_pr_preselected = go.Figure()
        for name, (precision, recall) in pr_data_preselected.items():
            fig_pr_preselected.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name))
        fig_pr_preselected.update_layout(title='Precision-Recall Curves (Preselected Features)')
        st.plotly_chart(fig_pr_preselected)

        st.subheader("ROC Curves Analysis")
        st.write("""
            The ROC (Receiver Operating Characteristic) curve is a crucial tool for evaluating the performance of classification models.
    
            - **Understanding ROC Curves**: This graph shows the trade-off between the true positive rate (sensitivity) 
              and the false positive rate (1-specificity) at various threshold settings. The area under the curve (AUC) 
              provides a single measure of a model's effectiveness. A higher AUC indicates a model with better 
              discriminative ability.
    
            - **Interpreting the Curve**: The closer the curve follows the left-hand border and then the top border 
              of the ROC space, the more accurate the test. A curve near the 45-degree diagonal represents a model 
              with no discriminatory power.
    
            ROC curves are particularly useful in settings with imbalanced classes and for comparing different models.
        """)        

        # Plotting ROC Curves for All Features
        fig_all = go.Figure()
        for name, (fpr, tpr, auc_score) in roc_data_all.items():
            fig_all.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc_score:.2f})'))
        fig_all.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
        fig_all.update_layout(title='ROC Curves (All Features)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig_all)

        # Plotting ROC Curves for Preselected Features
        fig_preselected = go.Figure()
        for name, (fpr, tpr, auc_score) in roc_data_preselected.items():
            fig_preselected.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc_score:.2f})'))
        fig_preselected.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
        fig_preselected.update_layout(title='ROC Curves (Preselected Features)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig_preselected)




# Tab 4: Model Evaluation with Cluster Features
with tab4:
    st.header("Model Improvement with K-Means Clustering")
    st.write("""
    This section explores the use of k-means clustering to enhance model accuracy. 
    By adding cluster labels as a feature, we investigate whether this additional 
    information can improve the predictive power of the models. You can select the 
    number of clusters and compare model performance with and without cluster features.
    """)
    st.header("Model Evaluation with Cluster Features")

    # Define classifiers with adjustable parameters
    n_neighbors = st.slider("Number of Neighbors for KNN", 1, 20, 5)
    n_estimators = st.slider("Number of Trees for Random Forest", 10, 100, 10)
    max_depth = st.slider("Max Depth for Random Forest", 1, 20, 5)

    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    }

    # Evaluate models on the original dataset
    st.subheader("Model Performance on Original Dataset")
    original_scores = evaluate_models_on_data(X, y, classifiers)
    # Evaluate models on the original dataset
    st.subheader("Model Performance on Original Dataset")
    original_scores = evaluate_models_on_data(X, y, classifiers)
    for model_name, accuracy in original_scores.items():
        st.write(f"{model_name} Accuracy (Original): {accuracy}")

    # PCA and Clustering Visualization
    st.subheader("PCA and Clustering Visualization")
    n_clusters_option = st.selectbox("Select number of clusters for K-Means:", [2, 3, 4], index=0)
    X_pca = perform_pca(X)
    cluster_labels = perform_kmeans(X_pca, n_clusters_option)
    plot_pca_clusters(X_pca, cluster_labels)

    # Button to add cluster labels to the dataset and evaluate models
    if st.button("Add Cluster Labels to Dataset and Evaluate"):
        X_with_clusters = np.hstack([X, cluster_labels.reshape(-1, 1)])
        st.session_state['X_augmented'] = X_with_clusters

        # Initialize dictionaries to store accumulated scores
        accumulated_scores_original = {model_name: 0 for model_name in classifiers.keys()}
        accumulated_scores_augmented = {model_name: 0 for model_name in classifiers.keys()}

        # Perform evaluation 10 times
        for _ in range(10):
            # Evaluate on original dataset
            scores_original = evaluate_models_on_data(X, y, classifiers)
            for model_name in scores_original:
                accumulated_scores_original[model_name] += scores_original[model_name]

            # Evaluate on augmented dataset
            scores_augmented = evaluate_models_on_data(st.session_state['X_augmented'], y, classifiers)
            for model_name in scores_augmented:
                accumulated_scores_augmented[model_name] += scores_augmented[model_name]

        # Calculate the average scores
        average_scores_original = {model_name: score / 10 for model_name, score in accumulated_scores_original.items()}
        average_scores_augmented = {model_name: score / 10 for model_name, score in accumulated_scores_augmented.items()}

        # Display the averaged scores
        st.subheader("Average Model Performance on Original Dataset")
        for model_name, accuracy in average_scores_original.items():
            st.write(f"{model_name} Average Accuracy (Original): {accuracy:.2%}")

        st.subheader("Average Model Performance on Dataset with Cluster Labels")
        for model_name, accuracy in average_scores_augmented.items():
            st.write(f"{model_name} Average Accuracy (With Clusters): {accuracy:.2%}")
            improvement = accuracy - average_scores_original[model_name]
            st.write(f"Average Accuracy Improvement: {improvement:.2%}")
        # Check if cluster labels are added to the dataset
        if 'X_augmented' in st.session_state:
            # Evaluate models on the augmented dataset
            st.subheader("Model Performance on Dataset with Cluster Labels")
            augmented_scores = evaluate_models_on_data(st.session_state['X_augmented'], y, classifiers)
            for model_name, accuracy in augmented_scores.items():
                st.write(f"{model_name} Accuracy (With Clusters): {accuracy}")
                improvement = accuracy - original_scores[model_name]
                st.write(f"Accuracy Improvement: {improvement:.2%}")

            # Comparative Visualization
            fig = px.bar(x=list(augmented_scores.keys()), 
                         y=[augmented_scores[model] - original_scores[model] for model in augmented_scores], 
                         labels={'x': 'Model', 'y': 'Accuracy Improvement'}, 
                         title="Model Accuracy Improvement with Cluster Labels")
            st.plotly_chart(fig)
