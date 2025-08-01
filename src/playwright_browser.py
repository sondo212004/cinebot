from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import time
import json

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


# ✅ Khai báo schema input dùng Pydantic
class CinemaShowtimesInput(BaseModel):
    specific_cinema_url: str = Field(..., description="URL trang lịch chiếu của rạp phim")
    name: str = Field(..., description="Tên rạp phim")
    location: str = Field(..., description="Địa điểm của rạp phim")
    source_url: Optional[str] = Field(None, description="URL nguồn (thường giống với specific_cinema_url)")


# ✅ Hàm trích xuất HTML
def extract_schedules_from_html(soup, date_str):
    movies_data = []
    label_blocks = soup.select('.film-label')
    print(f"🔎 Tìm thấy {len(label_blocks)} phim (film-label).")

    for label_div in label_blocks:
        try:
            title_elem = label_div.select_one('h3 a')
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)

            film_right_div = label_div.find_next_sibling('div', class_='film-right')
            if not film_right_div:
                continue

            showtime_elems = film_right_div.select('.film-showtimes li.item a span')
            showtimes = []
            for elem in showtime_elems:
                time_text = elem.get_text(strip=True)
                found_time = re.search(r'\d{1,2}:\d{2}', time_text)
                if found_time:
                    showtimes.append(found_time.group(0))

            if showtimes:
                unique_showtimes = sorted(list(set(showtimes)))
                movies_data.append({
                    'title': title,
                    'date': date_str,
                    'showtimes': unique_showtimes
                })
                print(f"  🎬 {title}: {', '.join(unique_showtimes)}")
        except Exception as e:
            print(f"  ⚠️ Lỗi khi xử lý phim: {e}")
            continue

    return movies_data


# ✅ Tool chính (hàm phải nhận 1 đối tượng Pydantic)
def scrape_cinema_showtimes_playwright(input_data: CinemaShowtimesInput) -> dict:
    """
    Lấy lịch chiếu phim từ một trang rạp cụ thể (CGV, Lotte...) trong 5 ngày tới.

    Args:
        input_data (CinemaShowtimesInput): Thông tin bao gồm URL, tên, vị trí rạp và nguồn.

    Returns:
        dict: Bao gồm thông tin rạp, thời gian scrape và lịch chiếu phim.
    """
    all_schedules = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        try:
            print(f"\n🚀 Bắt đầu scraping cho: {input_data.name}")
            print(f"🔗 URL: {input_data.specific_cinema_url}")
            page.goto(input_data.specific_cinema_url, timeout=15000)
            page.wait_for_selector("body", timeout=5000)

            for i in range(5):
                current_date = datetime.now() + timedelta(days=i)
                date_str = current_date.strftime('%Y%m%d')
                display_date = current_date.strftime('%Y-%m-%d')
                print(f"\n🗓️ Đang xử lý ngày: {display_date}")

                if i > 0:
                    try:
                        cgv_date_id = f"cgv{date_str}"
                        selector = f"#{cgv_date_id}"
                        page.click(selector, timeout=5000)
                        print(f"  ✅ Clicked tab ngày bằng ID: {cgv_date_id}")
                        time.sleep(2)
                    except Exception:
                        print("  ⚠️ Không tìm thấy tab ngày để click.")
                        break

                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                schedules_for_date = extract_schedules_from_html(soup, date_str)

                if schedules_for_date:
                    all_schedules.extend(schedules_for_date)
                else:
                    print("  ❌ Không tìm thấy lịch chiếu cho ngày này.")

            final_movies = {}
            for schedule in all_schedules:
                title = schedule['title']
                if title not in final_movies:
                    final_movies[title] = {'dates': {}}
                date = schedule['date']
                final_movies[title]['dates'][date] = schedule['showtimes']

            result_list = [{'title': title, **data} for title, data in final_movies.items()]
            print(f"\n🎉 Hoàn thành! Tìm thấy {len(result_list)} phim có lịch chiếu.")

            return {
                'status': 'success',
                'cinema_info': {
                    'name': input_data.name,
                    'location': input_data.location,
                    'source_url': input_data.source_url or input_data.specific_cinema_url
                },
                'scrape_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'schedules': result_list
            }

        except Exception as e:
            print(f"❌ Lỗi trong quá trình scraping: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'cinema_info': {
                    'name': input_data.name,
                    'location': input_data.location,
                    'source_url': input_data.source_url or input_data.specific_cinema_url
                }
            }
        finally:
            browser.close()
            print("🔐 Đã đóng trình duyệt.")


# ✅ Định nghĩa StructuredTool tương thích LangGraph Agent
cinema_showtimes_tool = StructuredTool.from_function(
    name="ScrapeCinemaShowtimes",
    func=scrape_cinema_showtimes_playwright,
    description=(
        "Dùng để lấy lịch chiếu phim từ một rạp cụ thể trong 5 ngày tới. "
        "Cần cung cấp URL của trang lịch chiếu và thông tin rạp (tên, địa chỉ). "
        "Trả về danh sách các bộ phim, thời gian chiếu và thông tin rạp."
    ),
    args_schema=CinemaShowtimesInput
)

# ✅ Test nếu chạy độc lập
if __name__ == '__main__':
    input_data = CinemaShowtimesInput(
        specific_cinema_url='https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/',
        name='CGV Vincom Center Bà Triệu',
        location='Hà Nội',
        source_url='https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/'
    )

    result = scrape_cinema_showtimes_playwright(input_data)
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*50)
    print(json.dumps(result, indent=4, ensure_ascii=False))
