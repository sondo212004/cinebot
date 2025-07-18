import re
import time
import traceback
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from langchain.tools import StructuredTool
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

def setup_selenium_driver():
    """Khởi tạo Selenium Chrome driver với các tùy chọn tối ưu."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Bỏ comment để chạy ẩn
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Tùy chọn để tăng tốc độ load
    prefs = {"profile.managed_default_content_settings.images": 2 # Không load ảnh
                , "profile.managed_default_content_settings.stylesheets": 2 # Không load CSS
             } 
    chrome_options.add_experimental_option("prefs", prefs)

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt WebDriver: {e}. Vui lòng kiểm tra kết nối mạng hoặc phiên bản Chrome.")
        return None

def extract_schedules_from_html(soup, date_str):
    movies_data = []

    label_blocks = soup.select('.film-label')  # Bắt đầu từ khối tiêu đề phim
    print(f"🔎 Tìm thấy {len(label_blocks)} phim (film-label).")

    for label_div in label_blocks:
        try:
            # Tìm tên phim
            title_elem = label_div.select_one('h3 a')
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)

            # Tìm div.film-right kế tiếp
            film_right_div = label_div.find_next_sibling('div', class_='film-right')
            if not film_right_div:
                continue

            # Tìm tất cả các suất chiếu
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



def scrape_cinema_showtimes(specific_cinema_url, cinema_info):
    """
    Tool chính: Scrape lịch chiếu từ một URL cụ thể của rạp.

    Args:
        specific_cinema_url (str): Link trực tiếp đến trang lịch chiếu của rạp.
        cinema_info (dict): Thông tin về rạp (tên, địa chỉ,...) để trả về trong kết quả.
    
    Returns:
        dict: Dữ liệu lịch chiếu hoặc thông báo lỗi.
    """
    driver = setup_selenium_driver()
    if not driver:
        return {'status': 'error', 'message': 'Không thể khởi tạo WebDriver.'}
        
    all_schedules = []
    
    try:
        print(f"\n🚀 Bắt đầu scraping cho: {cinema_info.get('name', specific_cinema_url)}")
        print(f"🔗 URL: {specific_cinema_url}")
        
        driver.get(specific_cinema_url)
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        # Lặp qua 5 ngày tới
        for i in range(2):
            current_date = datetime.now() + timedelta(days=i)
            date_str = current_date.strftime('%Y%m%d')
            display_date = current_date.strftime('%Y-%m-%d')
            
            print(f"\n🗓️ Đang xử lý ngày: {display_date}")

            # === SỬA LỖI TẠI ĐÂY ===
            # 1. Khởi tạo biến date_clicked = False
            date_clicked = False
            if i > 0: # Chỉ thử click từ ngày thứ hai trở đi
                try:
                    # ⭐ ƯU TIÊN SỬ DỤNG ID CỦA CGV
                    cgv_date_id = f"cgv{date_str}" 
                    
                    date_element = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.ID, cgv_date_id))
                    )
                    driver.execute_script("arguments[0].click();", date_element)
                    
                    # 2. Cập nhật biến nếu click thành công
                    date_clicked = True
                    print(f"  ✅ Clicked tab ngày bằng ID: {cgv_date_id}")
                    time.sleep(2) # Đợi AJAX load lại lịch chiếu

                except Exception:
                    # Thử các selector dự phòng khác ở đây nếu cần
                    print("  ⚠️ Không tìm thấy tab ngày để click.")
                    # Nếu không tìm thấy tab cho ngày tiếp theo, dừng lại
                    # vì trang web có thể không hỗ trợ xem nhiều ngày
                    break

            # Lấy HTML hiện tại sau khi đã click (hoặc không)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            schedules_for_date = extract_schedules_from_html(soup, date_str)
            
            if schedules_for_date:
                all_schedules.extend(schedules_for_date)
            else:
                print("  ❌ Không tìm thấy lịch chiếu cho ngày này.")
            
            # Nếu là lần lặp đầu tiên và không có tab nào được click,
            # giả định rằng trang chỉ hiển thị một ngày và dừng lại.
            # Logic này đã được xử lý bằng lệnh `break` ở trên.
        
        # Gom nhóm lại kết quả cuối cùng
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
            'schedules': result_list
        }
    

    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng trong quá trình scraping: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e),
            'cinema_info': cinema_info
        }
    finally:
        if driver:
            driver.quit()
            print("🔒 Đã đóng trình duyệt.")

cinema_showtimes_tool = StructuredTool.from_function(
    name="ScrapeCinemaShowtimes",
    func=scrape_cinema_showtimes,
    description=(
        "Dùng để lấy lịch chiếu phim từ một rạp cụ thể. "
        "Cần cung cấp URL của trang lịch chiếu và thông tin rạp (tên, địa chỉ). "
        "Trả về danh sách các bộ phim, thời gian chiếu và thông tin rạp."
        "\n\n"
        "Ví dụ: `ScrapeCinemaShowtimes(specific_cinema_url='https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/', cinema_info={'name': 'CGV Vincom Center Bà Triệu', 'location': 'Hà Nội', 'source_url': 'https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/'})`."
    )
)