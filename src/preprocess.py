import kagglehub
import pandas as pd
import os
import json
from langchain_core.documents import Document

# Download latest version
path = kagglehub.dataset_download("jrobischon/wikipedia-movie-plots")
csv_file = os.path.join(path, "wiki_movie_plots_deduped.csv")

df = pd.read_csv(csv_file)
df_5percent = df.sample(frac=0.05, random_state=42)
df_5percent.rename(columns={"Release Year": "Release_year","Origin/Ethnicity":"Nation","Wiki Page":"Wiki_Page"}, inplace=True)

movies = []

for movie in df_5percent.itertuples():
    movies.append({
        "Title": movie.Title,
        "Release_year": movie.Release_year,
        "Nation": movie.Nation,
        "Director": movie.Director,
        "Cast": movie.Cast,
        "Genre": movie.Genre,
        "Wiki_page": movie.Wiki_Page,
        "Plot": movie.Plot
    })



# Lưu trữ các tài liệu đã tạo
with open('movies.json', 'w', encoding='utf-8') as file:
  json.dump(movies, file, ensure_ascii=False, indent=2)