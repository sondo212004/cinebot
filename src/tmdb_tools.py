import os
import requests
import json
from typing import Type, Literal, List

# LangChain and Pydantic imports
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)  

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in environment variables. Please add it to your .env file.")

TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
DEFAULT_LANGUAGE = "vi-VN" # Default to Vietnamese
DEFAULT_REGION = "VN"     # Default to Vietnam region

def _format_json_output(data: dict, indent: int = 2) -> str:
    """Formats dictionary to a clean JSON string for LLM consumption."""
    return json.dumps(data, ensure_ascii=False, indent=indent)

# --- Input Schemas for Tools ---

class SearchInput(BaseModel):
    """Input for search tools."""
    query: str = Field(description="The search term, e.g., a movie title or a person's name.")

class DetailsInput(BaseModel):
    """Input for getting details by ID."""
    item_id: int = Field(description="The unique TMDB ID for the movie or person.")

class TrendingInput(BaseModel):
    """Input for the get_trending_movies tool."""
    time_window: Literal["day", "week"] = Field(
        default="day",
        description="The time window to get trending items for. Can be 'day' or 'week'."
    )

# --- TMDB LangChain Tools ---

@tool(args_schema=SearchInput)
def tmdb_movie_search(query: str) -> str:
    """
    Searches for movies by title. Returns a list of movies with their TMDB IDs, titles, and release dates.
    Use this tool to find the ID of a movie before getting its details.
    """
    endpoint = f"{TMDB_API_BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": DEFAULT_LANGUAGE}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "Không tìm thấy bộ phim nào với tên đó."

        # Simplify output for the agent
        simplified_results = [
            {"id": movie.get("id"), "title": movie.get("title"), "release_date": movie.get("release_date")}
            for movie in results[:5] # Return top 5 results
        ]
        return _format_json_output(simplified_results)
    except requests.RequestException as e:
        return f"Lỗi API khi tìm kiếm phim: {e}"

@tool(args_schema=DetailsInput)
def tmdb_get_movie_details(item_id: int) -> str:
    """
    Gets detailed information for a specific movie using its TMDB ID.
    Includes overview, genres, cast, director, and ratings.
    """
    endpoint = f"{TMDB_API_BASE_URL}/movie/{item_id}"
    params = {"api_key": TMDB_API_KEY, "language": DEFAULT_LANGUAGE, "append_to_response": "credits,reviews"}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract and simplify relevant information
        director = next((member['name'] for member in data.get('credits', {}).get('crew', []) if member['job'] == 'Director'), 'N/A')
        cast = [actor['name'] for actor in data.get('credits', {}).get('cast', [])[:5]]
        genres = [genre['name'] for genre in data.get('genres', [])]

        # Safely extract review snippet to prevent IndexError
        reviews_results = data.get('reviews', {}).get('results', [])
        if reviews_results and reviews_results[0].get('content'):
            review_snippet = reviews_results[0]['content'][:300] + "..."
        else:
            review_snippet = 'No reviews available.'

        details = {
            "id": data.get("id"),
            "title": data.get("title"),
            "overview": data.get("overview"),
            "release_date": data.get("release_date"),
            "vote_average": data.get("vote_average"),
            "genres": genres,
            "director": director,
            "cast": cast,
            "review_snippet": review_snippet
        }
        return _format_json_output(details)
    except requests.RequestException as e:
        return f"Lỗi API khi lấy chi tiết phim: {e}"


@tool(args_schema=SearchInput)
def tmdb_person_search(query: str) -> str:
    """
    Searches for people (actors, directors, etc.) by name.
    Returns a list of people with their TMDB IDs and known-for departments.
    """
    endpoint = f"{TMDB_API_BASE_URL}/search/person"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": DEFAULT_LANGUAGE}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "Không tìm thấy người nào với tên đó."

        simplified_results = [
            {"id": person.get("id"), "name": person.get("name"), "known_for_department": person.get("known_for_department")}
            for person in results[:5]
        ]
        return _format_json_output(simplified_results)
    except requests.RequestException as e:
        return f"Lỗi API khi tìm kiếm người: {e}"

@tool(args_schema=DetailsInput)
def tmdb_get_person_details(item_id: int) -> str:
    """
    Gets detailed information for a specific person using their TMDB ID.
    Includes biography and a list of notable movies they've been in.
    """
    endpoint = f"{TMDB_API_BASE_URL}/person/{item_id}"
    params = {"api_key": TMDB_API_KEY, "language": DEFAULT_LANGUAGE, "append_to_response": "movie_credits"}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        notable_movies = [
            {"title": movie.get("title"), "character": movie.get("character", "N/A")}
            for movie in sorted(data.get('movie_credits', {}).get('cast', []), key=lambda x: x.get('popularity', 0), reverse=True)[:5]
        ]

        details = {
            "name": data.get("name"),
            "biography": data.get("biography", "No biography available.")[:500] + "...",
            "birthday": data.get("birthday"),
            "place_of_birth": data.get("place_of_birth"),
            "known_for": notable_movies
        }
        return _format_json_output(details)
    except requests.RequestException as e:
        return f"Lỗi API khi lấy chi tiết người: {e}"

@tool
def tmdb_now_playing_movies() -> str:
    """
    Gets a list of movies currently playing in theaters in the default region (Vietnam).
    """
    endpoint = f"{TMDB_API_BASE_URL}/movie/now_playing"
    params = {"api_key": TMDB_API_KEY, "language": DEFAULT_LANGUAGE, "region": DEFAULT_REGION}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "Không có phim nào đang chiếu tại rạp."

        simplified_results = [{"id": movie.get("id"), "title": movie.get("title")} for movie in results[:10]]
        return _format_json_output(simplified_results)
    except requests.RequestException as e:
        return f"Lỗi API khi lấy phim đang chiếu: {e}"

@tool
def tmdb_upcoming_movies() -> str:
    """
    Gets a list of upcoming movies scheduled to be released in the default region (Vietnam).
    """
    endpoint = f"{TMDB_API_BASE_URL}/movie/upcoming"
    params = {"api_key": TMDB_API_KEY, "language": DEFAULT_LANGUAGE, "region": DEFAULT_REGION}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "Chưa có thông tin về phim sắp chiếu."

        simplified_results = [{"id": movie.get("id"), "title": movie.get("title"), "release_date": movie.get("release_date")} for movie in results[:10]]
        return _format_json_output(simplified_results)
    except requests.RequestException as e:
        return f"Lỗi API khi lấy phim sắp chiếu: {e}"

# List of all tools to be imported by the agent
tmdb_tools = [
    tmdb_movie_search,
    tmdb_get_movie_details,
    tmdb_person_search,
    tmdb_get_person_details,
    tmdb_now_playing_movies,
    tmdb_upcoming_movies,
]

