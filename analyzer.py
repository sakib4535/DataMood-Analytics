import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, chi2_contingency
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(
    page_title="DataMood Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define a custom banner
st.markdown(
    """
    <style>
        .main-container {
            background-color: #f4f4f4;
            padding: 2rem;
        }
        .banner {
            background-color: #009688;
            padding: 1rem;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='banner'>DataMood Explorer</h1>", unsafe_allow_html=True)


class EmotionAnalyzer:
    def __init__(self):
        self.emotional_data = pd.DataFrame()

    def load_emotional_data(self, file_path):
        self.emotional_data = pd.read_excel(file_path)

    def display_emotions(self, num_rows):
        st.write(f"Emotional Data (Showing {num_rows} rows):")
        st.table(self.emotional_data.head(num_rows))

    def analyze_emotions(self, sorting_algorithm="None", selected_features=[]):
        if sorting_algorithm != "None":
            self.sort_emotions(sorting_algorithm)
        else:
            self.emotional_data = self.emotional_data.sort_index()

        if selected_features:
            self.filter_features(selected_features)

    def sort_emotions(self, sorting_algorithm):
        if sorting_algorithm != "None":
            self.emotional_data = self.emotional_data.sort_values(by=self.emotional_data.columns[0])

    def filter_features(self, selected_features):
        selected_features = [self.emotional_data.columns[0]] + selected_features
        self.emotional_data = self.emotional_data[selected_features]

    def calculate_statistics(self):
        numeric_columns = self.emotional_data.select_dtypes(include=np.number).columns
        stats_dict = {
            "Mean": self.emotional_data.mean(),
            "Median": self.emotional_data.median(),
            "Mode": self.emotional_data.mode().iloc[0],
            "STD": self.emotional_data[numeric_columns].std(),
            "Skew": skew(self.emotional_data[numeric_columns], nan_policy='omit'),
            "Kurtosis": kurtosis(self.emotional_data[numeric_columns], nan_policy='omit'),
            "Variance": self.emotional_data.var(),
            "Chi-square Test": self.chi_square_test(),
            "Correlation": self.emotional_data[numeric_columns].corr(method='pearson'),
        }
        return stats_dict

    # Inside the EmotionAnalyzer class

    # Inside the EmotionAnalyzer class

    def chi_square_test(self):
        column_to_cut = self.emotional_data.iloc[:, 1]

        # Convert the column to numeric, coercing errors to NaN
        numeric_column = pd.to_numeric(column_to_cut, errors='coerce')

        # Drop rows with NaN values (non-numeric)
        numeric_column = numeric_column.dropna()

        # Check if there are values in the numeric column
        if not numeric_column.empty:
            # Apply cut to the numeric column
            categorical_data = pd.cut(numeric_column, bins=3, labels=["Low", "Medium", "High"])

            contingency_table = pd.crosstab(categorical_data, columns="count")
            chi2, p, _, _ = chi2_contingency(contingency_table)
            return {"Chi-square": chi2, "P-value": p}
        else:
            return {"Chi-square": None, "P-value": None}

    def generate_line_plot(self, x_column, y_column):
        fig, ax = plt.subplots()
        sorted_data = self.emotional_data.sort_values(by=y_column, ascending=False).head(10)
        ax.plot(sorted_data[x_column], sorted_data[y_column], marker='o')
        ax.set_title(f'{y_column} vs {x_column} (Top 10)')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')  # Rotate X-axis labels
        st.pyplot(fig)

    def generate_bar_plot(self, x_column, y_column):
        fig, ax = plt.subplots()
        sorted_data = self.emotional_data.sort_values(by=y_column, ascending=False).head(10)
        ax.bar(sorted_data[x_column], sorted_data[y_column])
        ax.set_title(f'{y_column} by {x_column} (Top 10)')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')  # Rotate X-axis labels
        st.pyplot(fig)


def main():
    st.title("Emotion Analyzer App")

    emotion_analyzer = EmotionAnalyzer()

    st.sidebar.title("Settings")

    # Allow user to upload an Excel file
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        emotion_analyzer.load_emotional_data(uploaded_file)

    sorting_algorithm = st.sidebar.selectbox("Select Sorting Algorithm", ["None", "bubble_sort", "insertion_sort", "selection_sort"])

    # Dynamically update available features based on the loaded Excel file columns
    feature_options = ["None"] + list(emotion_analyzer.emotional_data.columns[1:])
    selected_features = st.sidebar.multiselect("Select Emotional Features", feature_options)

    # Slider for selecting number of rows
    num_rows = st.sidebar.slider("Select Number of Rows to Display", 1, len(emotion_analyzer.emotional_data), 10)

    emotion_analyzer.analyze_emotions(sorting_algorithm, selected_features)
    emotion_analyzer.display_emotions(num_rows)

    if st.button("Analyze and Calculate Statistics"):
        stats_dict = emotion_analyzer.calculate_statistics()
        st.write("\n**Statistical Analysis:**")
        for stat_name, stat_values in stats_dict.items():
            st.write(f"\n{stat_name.capitalize()}:")
            if isinstance(stat_values, dict):
                st.table(pd.DataFrame(stat_values, index=["Value"]))
            else:
                st.table(stat_values)

    # Line graph
    if st.sidebar.checkbox("Line Graph"):
        st.sidebar.subheader("Line Graph Settings")
        x_axis_line = st.sidebar.selectbox("Select X-axis column", emotion_analyzer.emotional_data.columns)
        y_axis_line = st.sidebar.selectbox("Select Y-axis column", emotion_analyzer.emotional_data.columns)
        emotion_analyzer.generate_line_plot(x_axis_line, y_axis_line)

    # Bar graph
    if st.sidebar.checkbox("Bar Graph"):
        st.sidebar.subheader("Bar Graph Settings")
        x_axis_bar = st.sidebar.selectbox("Select X-axis column", emotion_analyzer.emotional_data.columns)
        y_axis_bar = st.sidebar.selectbox("Select Y-axis column", emotion_analyzer.emotional_data.columns)
        emotion_analyzer.generate_bar_plot(x_axis_bar, y_axis_bar)


if __name__ == "__main__":
    main()
