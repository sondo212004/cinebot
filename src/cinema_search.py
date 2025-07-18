import requests
from langchain.tools import Tool

def find_nearest_cinemas(user_location: str):
    """
    Tìm kiếm các rạp chiếu phim gần một địa điểm do người dùng cung cấp bằng cách sử dụng OpenStreetMap và Nominatim API.

    Args:
        user_location: Địa điểm do người dùng nhập vào (ví dụ: "Hồ Gươm, Hà Nội").

    Returns:
        Một danh sách các rạp chiếu phim gần đó hoặc một thông báo lỗi.
    """
    headers = {
        'User-Agent': 'MovieChatbotProject/1.0 (sondo212004@gmail.com)' 
    }

    # Bước 1: Lấy tọa độ (latitude, longitude) từ địa chỉ người dùng bằng Nominatim API
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': user_location,
        'format': 'json',
        'limit': 1
    }
    try:
        response = requests.get(nominatim_url, params=params, headers=headers,timeout=20)
        response.raise_for_status()
        location_data = response.json()
        if not location_data:
            return "Không thể tìm thấy địa điểm của bạn. Vui lòng thử lại với một địa chỉ khác."

        lat = location_data[0]['lat']
        lon = location_data[0]['lon']

    except requests.exceptions.RequestException as e:
        # Lỗi sẽ không còn là 403 Forbidden sau khi thêm User-Agent
        return f"Lỗi khi kết nối đến Nominatim API: {e}"

    # Bước 2: Tìm kiếm rạp chiếu phim gần tọa độ đã cho bằng Overpass API
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node(around:5000,{lat},{lon})[amenity=cinema];
      way(around:5000,{lat},{lon})[amenity=cinema];
      relation(around:5000,{lat},{lon})[amenity=cinema];
    );
    out body;
    >;
    out skel qt;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers,timeout=20)
        response.raise_for_status()
        cinema_data = response.json()

        cinemas = []
        if cinema_data.get('elements'):
            for element in cinema_data['elements']:
                if 'tags' in element and 'name' in element['tags']:
                    cinema_name = element['tags']['name']
                    cinemas.append(cinema_name)

        if not cinemas:
            return f"Không tìm thấy rạp chiếu phim nào trong vòng 5km quanh '{user_location}'."

        # Loại bỏ các rạp trùng lặp và trả về kết quả
        unique_cinemas = sorted(list(set(cinemas)))
        return f"Các rạp chiếu phim gần '{user_location}':\n- " + "\n- ".join(unique_cinemas)

    except requests.exceptions.RequestException as e:
        return f"Lỗi khi kết nối đến Overpass API: {e}"

# Định nghĩa Tool cho LangChain
cinema_search_tool = Tool(
    name="CinemaSearch",
    func=find_nearest_cinemas,
    description=(
        "Tìm kiếm các rạp chiếu phim gần một địa điểm do người dùng cung cấp. "
        "Trả về danh sách các rạp chiếu phim gần đó."
    )
)