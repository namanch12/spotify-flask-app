from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load full original dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Load the same sampled dataset used during training (30000 rows)
data = pd.read_csv("sampled_spotify_data.csv")

# Features used for clustering
features = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
            'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Load scaler and model
scaler = joblib.load("scaler_spotify.pkl")
kmeans = joblib.load("kmeans_spotify.pkl")

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    song_features = None
    song_name = ""

    if request.method == 'POST':
        song_name = request.form['song_name'].strip()

        # Find the song in full dataset
        row = df[df['track_name'].str.lower() == song_name.lower()]

        if row.empty:
            recommendations = ["❌ Song not found. Please try another."]
        else:
            input_index = row.index[0]
            scaled_input = scaler.transform(row[features])
            target_cluster = kmeans.predict(scaled_input)[0]

            # Filter same cluster songs from the sampled dataset
            data['Cluster'] = kmeans.labels_.astype(str)
            similar_songs = data[data['Cluster'] == str(target_cluster)]

            recs = similar_songs.sample(n=5, random_state=42)['track_name'].tolist()
            recommendations = recs

            # Get the audio features of the searched song
            song_features = row[features].to_dict(orient='records')[0]

    return render_template("index.html",
                           recommendations=recommendations,
                           song_features=song_features,
                           song_name=song_name)

if __name__ == '__main__':
    app.run(debug=True)