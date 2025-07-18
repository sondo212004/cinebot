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


documents = []
for movie in movies:
  # Tạo nội dung văn bản cho embedding
  content = (
    f"Title: {movie.get('Title', 'Unknown')}\n"
    f"Release Year: {movie.get('Release_year', 'Unknown')}\n"
    f"Origin/Ethnicity: {movie.get('Nation', 'Unknown')}\n"
    f"Director: {movie.get('Director', 'Unknown')}\n"
    f"Cast: {movie.get('Cast', 'Unknown')}\n"
    f"Genre: {movie.get('Genre', 'Unknown')}\n"
    f"Plot: {movie.get('Plot', 'Unknown')}\n"
  )

  # Metadata để lưu trữ thông tin chi tiết
  metadata = {
    "Title": movie.get('Title', 'Unknown'),
    "Release_year": movie.get('Release_year', 'Unknown'),
    "Nation": movie.get('Nation', 'Unknown'),
    "Director": movie.get('Director', 'Unknown'),
    "Cast": movie.get('Cast', 'Unknown'),
    "Genre": movie.get('Genre', 'Unknown'),
    "Wiki_page": movie.get('Wiki_page', 'Unknown'),
    "Plot": movie.get('Plot', 'Unknown')
  }

  documents.append(Document(page_content=content, metadata=metadata))

# Lưu trữ các tài liệu đã tạo
with open('movies.json', 'w', encoding='utf-8') as file:
  json.dump(movies, file, ensure_ascii=False, indent=2)