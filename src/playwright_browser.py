import re
import traceback
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from langchain.tools import StructuredTool
from playwright.sync_api import sync_playwright
from concurrent.futures import ProcessPoolExecutor

def setup_playwright_browser(playwright):
    """Khởi tạo trình duyệt Chromium của Playwright với các tùy chọn tối ưu."""
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--start-maximized",
            "--disable-blink-features=AutomationControlled",
        ],
    )

    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
        extra_http_headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        },
    )

    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    return browser, context


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
                movies_data.append({'title': title, 'date': date_str, 'showtimes': unique_showtimes})
                print(f"  🎬 {title}: {', '.join(unique_showtimes)}")
        except Exception as e:
            print(f"  ⚠️ Lỗi khi xử lý phim: {e}")
            continue

    return movies_data


def scrape_cinema_showtimes(specific_cinema_url: str, cinema_info: dict = None):
    """
    Công cụ chính: Lấy lịch chiếu phim từ một URL cụ thể của rạp.
    Trả về dict với trạng thái success/error và dữ liệu schedules.
    """
    cinema_info = cinema_info or {}

    with sync_playwright() as playwright:
        browser, context = setup_playwright_browser(playwright)
        page = context.new_page()

        all_schedules = []

        try:
            print(f"\n🚀 Bắt đầu scraping cho: {cinema_info.get('name', specific_cinema_url)}")
            print(f"🔗 URL: {specific_cinema_url}")

            page.goto(specific_cinema_url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(2000)

            for i in range(5):
                current_date = datetime.now() + timedelta(days=i)
                date_str = current_date.strftime('%Y%m%d')
                display_date = current_date.strftime('%Y-%m-%d')

                print(f"\n🗓️ Đang xử lý ngày: {display_date}")

                if i > 0:
                    try:
                        cgv_date_id = f"cgv{date_str}"
                        date_element = page.locator(f"#{cgv_date_id}")
                        date_element.wait_for(state="visible", timeout=5000)
                        date_element.click()
                        print(f"  ✅ Đã click tab ngày bằng ID: {cgv_date_id}")
                        page.wait_for_timeout(2000)
                    except Exception:
                        print("  ⚠️ Không tìm thấy tab ngày để click.")

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
                'cinema_info': cinema_info,
                'scrape_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'schedules': result_list,
            }

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong quá trình scraping: {e}")
            traceback.print_exc()
            return {'status': 'error', 'message': str(e), 'cinema_info': cinema_info}
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
            print("🔒 Đã đóng trình duyệt.")

def scrape_cinema_showtimes_wrapper(specific_cinema_url, cinema_info=None):
    """Wrapper để chạy scrape_cinema_showtimes trong ProcessPoolExecutor"""
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(scrape_cinema_showtimes, specific_cinema_url, cinema_info)
        return future.result()

cinema_showtimes_tool = StructuredTool.from_function(
    name="ScrapeCinemaShowtimes",
    func=scrape_cinema_showtimes_wrapper,  # chạy sync
    description=(
        "Dùng để lấy lịch chiếu phim từ một rạp cụ thể. "
        "Cần cung cấp URL của trang lịch chiếu và thông tin rạp (tên, địa chỉ). "
        "Trả về danh sách các bộ phim, thời gian chiếu và thông tin rạp."
    ),
)


if __name__ == '__main__':
    example_url = "https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/"
    example_cinema_info = {
        'name': 'CGV Vincom Center Bà Triệu',
        'location': 'Hà Nội',
        'source_url': example_url,
    }

    result = scrape_cinema_showtimes(example_url, example_cinema_info)
    print(json.dumps(result, indent=2, ensure_ascii=False))
