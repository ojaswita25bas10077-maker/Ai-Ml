import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
data = {
    'title': [
        'Avengers', 'Iron Man', 'Captain America',
        'Thor', 'Hulk', 'Batman', 'Superman'
    ],
    'genre': [
        'action superhero', 'action tech', 'action patriotic',
        'action myth', 'action strong', 'dark action', 'action alien'
    ]
}

df = pd.DataFrame(data)

# Convert text to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

# Similarity matrix
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    if movie not in df['title'].values:
        return "Movie not found!"

    index = df[df['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i in movies:
        recommendations.append(df.iloc[i[0]].title)

    return recommendations

# Test
movie_name = input("Enter a movie: ")
print("Recommended Movies:", recommend(movie_name))
