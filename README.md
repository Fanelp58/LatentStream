# LatentStream

LatentStream is a real-time movie recommendation system built with Streamlit. It leverages a pre-trained Alternating Least Squares (ALS) model to provide instant movie suggestions based on user ratings. The application demonstrates a powerful cold-start strategy by dynamically calculating a latent vector for new users without retraining the entire model.

## Features

- **Real-time Recommendations:** Get movie recommendations instantly as you rate films.
- **Cold-Start Handling:** Employs a matrix factorization projection technique to generate personalized recommendations for new users on-the-fly.
- **Interactive UI:** A sleek, Netflix-inspired dark-mode dashboard built with Streamlit allows users to easily search for, rate, and discover movies.
- **ALS Model:** The recommendation engine is powered by the `implicit` library's implementation of Alternating Least Squares, trained on the MovieLens 1M dataset.
- **Performance Metrics:** The dashboard displays key technical indicators like inference latency and model parameters.

## How It Works

The system consists of two main parts: an offline training script and a live inference application.

1.  **Model Training (`MyALS.py`)**:
    *   The MovieLens 1M dataset is downloaded and preprocessed. User and movie IDs are mapped to matrix indices.
    *   An ALS model from the `implicit` library is trained on the user-item interaction matrix.
    *   The trained model (`als_model.npz`) and necessary metadata like ID mappings (`als_metadata.pkl`) are serialized to disk.

2.  **Live Inference (`app.py`)**:
    *   The Streamlit application loads the pre-trained ALS model and metadata.
    *   Users search for movies and provide ratings (from 1 to 5 stars).
    *   When the user requests recommendations, a sparse vector representing their ratings (`R_new`) is created.
    *   Instead of retraining, the system calculates a new user latent vector ($P_{new}$) by projecting their ratings onto the fixed item-factor matrix ($Q$):
        
        $P_{new} = (Q^T Q + \lambda I)^{-1} Q^T R_{new}$
        
    *   This calculation is handled efficiently by the `model.recommend(recalculate_user=True)` function.
    *   The top N recommendations are determined and displayed in the user interface, complete with movie posters fetched from the OMDb API.

## Technical Stack

- **Machine Learning:** `implicit`, `pandas`, `numpy`, `scipy`
- **Web Framework:** `streamlit`
- **Dataset:** MovieLens 1M

## Setup and Usage

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fanelp58/LatentStream.git
    cd LatentStream
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Key:**
    The application uses the OMDb API to fetch movie posters. You will need a free API key.
    
    *   Create a file named `.streamlit/secrets.toml`.
    *   Add your API key to the file like this:
        ```toml
        OMDB_API_KEY = "YOUR_API_KEY_HERE"
        ```

4.  **Run the Streamlit app:**
    The pre-trained model `als_model.npz` and `als_metadata.pkl` are included in the repository, so you do not need to run the training script.
    
    ```bash
    streamlit run app.py
    ```
    
    The application will open in your default web browser.

## File Descriptions

-   `app.py`: The main Streamlit web application for live recommendations.
-   `MyALS.py`: The script used for downloading data, training the ALS model, and serializing the artifacts.
-   `als_model.npz`: The serialized weights of the trained implicit ALS model.
-   `als_metadata.pkl`: A pickle file containing metadata required for the app, such as movie dataframes and ID mappings.
-   `requirements.txt`: A list of Python packages required to run the project.
