import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset (UPDATED)
data = {
    'title': [
        'Avengers', 'Iron Man', 'Captain America',
        'Thor', 'Hulk', 'Batman', 'Superman',
        'Inception', 'Interstellar', 'Dark Knight'
    ],
    'genre': [
        'action superhero', 'action tech', 'action patriotic',
        'action myth', 'action strong', 'dark action', 'action alien',
        'sci-fi thriller', 'space sci-fi', 'dark action crime'
    ]
}

df = pd.DataFrame(data)

# Convert text to numbers
cv = CountVectorizer()
matrix = cv.fit_transform(df['genre'])

# Similarity check
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    movie = movie.strip().lower()
    
    titles = df['title'].str.lower()
    
    if movie not in titles.values:
        return "❌ Movie not found! Try another one."
    
    index = titles[titles == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i in movies:
        recommendations.append(df.iloc[i[0]].title)

    return recommendations

# Run program
movie_name = input("🎬 Enter a movie name: ")

result = recommend(movie_name)

if isinstance(result, list):
    print("\n✨ Recommended Movies:")
    for i, movie in enumerate(result, 1):
        print(f"{i}. {movie}")
else:
    print(result)

# Test
movie_name = input("Enter a movie: ")
print("Recommended Movies:", recommend(movie_name))
