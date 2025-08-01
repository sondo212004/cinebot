# CineBot - Trợ Lý Phim Ảnh AI Toàn Diện

---

## Giới Thiệu Chung

**CineBot** là một trợ lý ảo thông minh được phát triển dựa trên kiến trúc **Agentic AI** mạnh mẽ của **LangGraph**, cung cấp cho người dùng khả năng truy vấn và nhận thông tin chi tiết về phim ảnh, diễn viên, đạo diễn, và đặc biệt là lịch chiếu phim tại các rạp cụ thể theo thời gian thực. Dự án này thể hiện sự kết hợp mạnh mẽ của các mô hình ngôn ngữ lớn (LLM) với khả năng tích hợp công cụ đa dạng, từ tìm kiếm web đến tự động hóa trình duyệt.

Với CineBot, người dùng có thể:

- Tìm kiếm thông tin tóm tắt, diễn viên, đạo diễn, thể loại của bất kỳ bộ phim nào.
- Khám phá các bộ phim đang chiếu và sắp ra mắt.
- Tra cứu các rạp chiếu phim gần vị trí của họ.
- **Xem lịch chiếu chi tiết của phim tại một rạp cụ thể trong 5 ngày tới.**

---

## Kiến Trúc và Công Nghệ Sử Dụng

CineBot được xây dựng trên một kiến trúc module, cho phép mở rộng và tích hợp dễ dàng các chức năng mới. Các công nghệ chính bao gồm:

- **Ngôn ngữ lập trình:** Python
- **Khung Agent:**
  - **LangGraph Agentic AI:** Điều phối toàn bộ luồng xử lý và ra quyết định của chatbot. LangGraph cung cấp một framework mạnh mẽ để định nghĩa các trạng thái, chuyển đổi giữa các trạng thái, và kiểm soát luồng xử lý của Agent một cách rõ ràng và có thể debug. Nó cho phép Agent tự động chọn và sử dụng các công cụ phù hợp dựa trên ngữ cảnh hội thoại và trạng thái hiện tại.
- **Mô hình Ngôn ngữ Lớn (LLM):** GPT-4o-mini (hoặc các mô hình tương đương của OpenAI)
- **Backend API:**
  - **FastAPI:** Xây dựng một API RESTful hiện đại, hiệu suất cao để giao tiếp với Chatbot Engine. Hỗ trợ cả phản hồi thông thường và phản hồi streaming (Server-Sent Events) cho trải nghiệm người dùng mượt mà hơn.
- **Các Công Cụ (Tools):**
  - **RAG (Retrieval Augmented Generation):** Sử dụng cơ sở tri thức nội bộ để cung cấp thông tin phim đã biết, giảm thiểu gọi API bên ngoài cho các truy vấn cơ bản.
  - **Tavily Search (Web Search Tool):** Tích hợp tìm kiếm web để thu thập thông tin cập nhật, tin tức, hoặc các liên kết quan trọng không có trong database. Đặc biệt được dùng để **tìm URL chính xác của các rạp chiếu phim** từ tên rạp và địa điểm.
  - **Nominatim (Cinema Search Tool):** Tìm kiếm và định vị các rạp chiếu phim dựa trên vị trí địa lý của người dùng, cung cấp danh sách các rạp tiềm năng.
  - **TMDB API Tools:** Tương tác với The Movie Database để lấy dữ liệu phim và người nổi tiếng (tóm tắt, diễn viên, đạo diễn, thể loại, ngày phát hành, v.v.).
  - **Playwright (Scrape Showtimes Tool):** Tự động hóa trình duyệt (chế độ headless) để **cạo dữ liệu lịch chiếu phim trực tiếp từ website của các hệ thống rạp** (hiện tại hỗ trợ CGV). Playwright cung cấp khả năng tương tác mạnh mẽ và đáng tin cậy với các trang web động, thay thế Selenium để cải thiện hiệu suất và độ ổn định.

---

## Tính Năng Nổi Bật

- **Kiến trúc Agent mạnh mẽ:** Sử dụng LangGraph để điều phối các tác vụ phức tạp và đa bước của chatbot.
- **Truy vấn thông tin phim:** Tóm tắt, diễn viên, đạo diễn, thể loại, năm phát hành.
- **Cập nhật phim đang chiếu/sắp chiếu:** Thông tin mới nhất về thị trường điện ảnh.
- **Tìm rạp gần bạn:** Dựa trên vị trí người dùng cung cấp.
- **Lịch chiếu phim chi tiết:** Hiển thị thời gian chiếu cụ thể của từng phim tại rạp đã chọn trong 5 ngày tới, được tổng hợp từ website của rạp.
- **API Backend hoàn chỉnh:** Cung cấp API RESTful cho phép tích hợp dễ dàng với các giao diện người dùng khác (web, mobile).
- **Phản hồi Streaming:** Hỗ trợ chế độ phản hồi theo luồng (stream) để hiển thị tin nhắn của bot một cách tức thì và mượt mà.
- **Tương tác thông minh:** Agent tự động điều phối các công cụ để trả lời các truy vấn phức tạp và đa bước.
- **Tối ưu hiệu suất:** Sử dụng chế độ headless cho Selenium và hạn chế các cuộc gọi không cần thiết để cải thiện thời gian phản hồi.

---

## Cài Đặt và Chạy Dự Án

Để cài đặt và chạy CineBot, bạn cần thực hiện các bước sau:

1.  **Clone Repository:**

    ```bash
    git clone [https://github.com/sondo212004/cinebot.git](https://github.com/sondo212004/cinebot.git)
    cd cinebot
    ```

2.  **Tạo và Kích hoạt Môi Trường Ảo:**

    ```bash
    python -m venv venv
    # Trên Windows
    .\venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài Đặt Các Thư Viện Cần Thiết:**

    ```bash
    pip install -r requirements.txt
    # Playwright cũng yêu cầu cài đặt trình duyệt
    playwright install
    ```

4.  **Cấu Hình Biến Môi Trường:**
    Tạo một file `.env` ở thư mục gốc của dự án và thêm các khóa API của bạn:

    ```
    OPENAI_API_KEY="your_openai_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    # Có thể thêm các biến khác nếu cần cho Nominatim/TMDB nếu chúng yêu cầu key
    ```

5.  **Chạy Backend API:**

    ```bash
    uvicorn api.main_api:app --reload --port 8000

    ```

6.  **Mở Giao Diện Người Dùng (Frontend):**
    Mở file ui/chatbot_ui.html trong trình duyệt web của bạn. Đây là giao diện để tương tác với CineBot thông qua API backend.

---

## Hướng Phát Triển Tương Lai

- Mở rộng hỗ trợ scraping cho các hệ thống rạp khác (ví dụ: Lotte Cinema, BHD Star).
- Tích hợp tính năng đặt vé trực tiếp hoặc cung cấp link đặt vé.
- Thử nghiệm với **CrewAI** để xây dựng hệ thống multi-agent, nơi các Agent chuyên biệt hóa các nhiệm vụ (ví dụ: Agent tìm kiếm, Agent phân tích, Agent tổng hợp).
- Xây dựng giao diện người dùng (UI) mạnh mẽ hơn (ví dụ: dùng React, Vue, Svelte) để nâng cao trải nghiệm.
- Tối ưu hóa hơn nữa tốc độ scraping và giảm thiểu chi phí.

---

## Liên Hệ

Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ qua:

- SĐT: 0961117894
- Email: [sondo212004@gmail.com]

---
