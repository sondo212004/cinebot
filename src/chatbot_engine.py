import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain core imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever, SystemMessage

# Import mới để tránh deprecation warning
try:
    from langchain_chroma import Chroma
    print("✅ Sử dụng langchain_chroma mới")
except ImportError:
    from langchain.vectorstores import Chroma
    print("⚠️ Sử dụng Chroma cũ")

# Import embeddings mới
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Sử dụng langchain_huggingface mới")
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ Sử dụng langchain_community.embeddings")
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings
        print("⚠️ Sử dụng HuggingFaceEmbeddings cũ")

# Giả lập web_search_tool nếu file không tồn tại để code có thể chạy độc lập
try:
    from web_search_agent import web_search_tool
    from cinema_search import cinema_search_tool
    from tmdb_tools import tmdb_tools
    from scrape_cinema_showtimes import cinema_showtimes_tool
except ImportError:
    print("⚠️ Không tìm thấy web_search_agent.py, cinema_search.py, tmdb_tools.py,tạo tool giả lập.")
    from langchain.tools import DuckDuckGoSearchRun
    web_search_tool = DuckDuckGoSearchRun()


class ChatbotEngine:
    """Engine xử lý logic chatbot với 1 phiên duy nhất xuyên suốt."""
    
    def __init__(self):
        # Cấu hình
        self.MODEL = "gpt-4o-mini"
        self.db_name = 'vector_db'

        self.SYSTEM_PROMPT = SystemMessage(content=""" 
    Bạn là CineBot - một chuyên gia tư vấn phim ảnh thông minh và thân thiện.

    NHIỆM VỤ CỐT LÕI:
    - Phân tích kỹ lưỡng yêu cầu của người dùng, ĐẶC BIỆT LƯU Ý đến lịch sử trò chuyện (`chat_history`) để hiểu rõ ngữ cảnh.
    - Nếu người dùng hỏi một câu nối tiếp (ví dụ: "còn phim nào khác không?", "phim đó của ai?"), bạn PHẢI dựa vào `chat_history` để suy ra chủ đề hoặc bộ phim đang được nói đến.

    **QUY TRÌNH TÌM KIẾM THÔNG TIN:**
    1. **ƯU TIÊN 1: Tìm kiếm trong cơ sở dữ liệu phim nội bộ (`movie_database_search`).**
        * Sử dụng công cụ `movie_database_search` cho các câu hỏi đơn giản và trực tiếp về tóm tắt phim, diễn viên, đạo diễn, thể loại, năm sản xuất có thể đã có trong dữ liệu bạn được huấn luyện.
    2. **ƯU TIÊN 2: Sử dụng các công cụ TMDB (`tmdb_*`) VÀ tìm kiếm web (`web_search_tool`).**
        * **Đây là nguồn thông tin ĐẦY ĐỦ và CHÍNH XÁC nhất về phim ảnh và người nổi tiếng.**
        * **Luôn ưu tiên các công cụ TMDB (`tmdb_movie_search`, `tmdb_get_movie_details`, `tmdb_person_search`, `tmdb_get_person_details`, `tmdb_now_playing_movies`, `tmdb_upcoming_movies`)** để lấy thông tin cơ bản.
        * **ĐỒNG THỜI, HÃY SỬ DỤNG `web_search_tool` (tavily_search) để xác minh hoặc tìm kiếm thông tin chi tiết hơn, ĐẶC BIỆT LÀ NGÀY PHÁT HÀNH CỤ THỂ TẠI VIỆT NAM (hoặc quốc gia người dùng quan tâm nếu biết).** Ví dụ, sau khi lấy ngày phát hành từ TMDB, hãy tìm kiếm "ngày phát hành [Tên phim] Việt Nam" bằng `web_search_tool` để đảm bảo thông tin chính xác nhất cho thị trường địa phương.
        * **NHỚ:** Nếu bạn chỉ có tên phim/người, hãy dùng `tmdb_movie_search` hoặc `tmdb_person_search` trước để lấy ID, sau đó dùng ID đó với `tmdb_get_movie_details` hoặc `tmdb_get_person_details` để lấy thông tin chi tiết.
    3. **ƯU TIÊN 3: Tìm kiếm web (`web_search_tool`) ĐỘC LẬP.**
        * Chỉ sử dụng công cụ `web_search_tool` (web search) khi thông tin KHÔNG CÓ trong database nội bộ hoặc TMDB, hoặc khi người dùng hỏi về các tin tức, sự kiện rất mới mà các nguồn khác không cập nhật kịp (ví dụ: "tin tức điện ảnh mới nhất", "sự kiện liên quan đến diễn viên [tên] gần đây").

    - Đưa ra gợi ý phim phù hợp kèm lý do thuyết phục.

    QUY TẮC GIAO TIẾP:
    - Trả lời bằng ngôn ngữ của người dùng (nếu có thể).
    - Giữ câu trả lời ngắn gọn, súc tích, tránh lặp lại
    - Giữ giọng văn thân thiện, gần gũi, sử dụng emoji để cuộc trò chuyện thêm sinh động.
    - Không bao giờ bịa đặt thông tin. Nếu không biết, hãy nói là không biết.

    ---

    ### QUY TẮC ĐẶC BIỆT KHI HỎI VỀ PHIM CHIẾU RẠP VÀ LỊCH CHIẾU CHI TIẾT:

    1.  Nếu người dùng hỏi về một bộ phim đang chiếu rạp và muốn biết lịch chiếu hoặc địa điểm xem, hãy sử dụng web_search_tool để tìm kiếm url của rạp chiếu đó và dùng cinema_scrape_tool để đưa ra lịch chiếu cho họ).
    2.  **Sau khi có vị trí (tỉnh/thành phố hoặc địa chỉ cụ thể), SỬ DỤNG CÔNG CỤ `cinema_search_tool` với `movie_name` và `location` để tìm danh sách các rạp đang chiếu phim đó ở gần người dùng.** Công cụ này sẽ trả về **TÊN RẠP** và **ĐỊA CHỈ**. Nếu input của người dùng đã có rạp chiếu phim cụ thể, hãy sử dụng luôn web_search_tool, không cần tìm những địa điểm gần đó.
    3.  **Từ kết quả của `cinema_search_tool`, HÃY CHỌN MỘT HOẶC HAI TÊN RẠP ĐẠI DIỆN** (ví dụ: "CGV ,Vincom Center Bà Triệu", "Lotte Cinema Royal City").
    4.  **VỚI MỖI TÊN RẠP ĐƯỢC CHỌN, HÃY SỬ DỤNG `web_search_tool` để tìm kiếm URL TRANG LỊCH CHIẾU CỤ THỂ CỦA RẠP ĐÓ.**
        * Ví dụ: Nếu `cinema_search_tool` trả về "CGV Vincom Center Bà Triệu", bạn sẽ gọi `web_search_tool("trang lịch chiếu CGV Vincom Center Bà Triệu")`.
        * **Sau đó, PHÂN TÍCH KẾT QUẢ TỪ `web_search_tool` để TRÍCH XUẤT URL TRANG LỊCH CHIẾU CHÍNH XÁC.** (Thường là link từ miền chính thức của rạp như cgv.vn, lottecinemavn.com, bhdstar.vn).
    5.  **Sau khi đã có `specific_cinema_url` (từ `web_search_tool`) và `cinema_info` (bao gồm tên rạp và địa điểm, lấy từ `cinema_search_tool`), HÃY SỬ DỤNG CÔNG CỤ `ScrapeCinemaShowtimes` để lấy lịch chiếu chi tiết.**
        * Bạn **PHẢI truyền đúng hai tham số**: `specific_cinema_url` và `cinema_info`.
        * Ví dụ gọi công cụ: `ScrapeCinemaShowtimes(specific_cinema_url='https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/', cinema_info={'name': 'CGV Vincom Center Bà Triệu', 'location': 'Hà Nội', 'source_url': 'https://www.cgv.vn/default/cinox/site/cgv-vincom-center-ba-trieu/'})`.
    6.  **Sau khi `ScrapeCinemaShowtimes` trả về dữ liệu lịch chiếu chi tiết (dạng dictionary), bạn HÃY PHÂN TÍCH DỮ LIỆU ĐÓ và TỔNG HỢP, SẮP XẾP, TRÌNH BÀY THÔNG TIN một cách rõ ràng, đầy đủ và thân thiện cho người dùng.**
    7.  Đảm bảo câu trả lời bao gồm tên phim, các rạp chiếu, thời gian chiếu cụ thể cho từng ngày, và đường dẫn để đặt vé nếu có.
    """        
    )

        

        # Khởi tạo components
        self.llm, self.retriever, self.tools = self._initialize_components()

        # Tạo agent duy nhất
        self.agent_executor = self._create_agent()

    def _load_environment(self):
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY không tìm thấy trong file .env!")
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key

    def _movie_search_function(self, query: str) -> str:
        print(f"🔎 Đang tìm kiếm trong database với query: '{query}'")
        try:
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                return "Không tìm thấy thông tin phim phù hợp trong cơ sở dữ liệu."
            result = "Thông tin phim tìm được từ database:\n\n"
            for i, doc in enumerate(docs[:3], 1):
                title = doc.metadata.get('title', f'Phim {i}')
                content = doc.page_content[:500]
                result += f"**{title}:**\n{content}\n\n"
            return result
        except Exception as e:
            return f"Lỗi khi tìm kiếm cơ sở dữ liệu: {str(e)}"

    def _initialize_components(self):
        print("🔧 Đang khởi tạo chatbot components...")
        self._load_environment()
        if not os.path.exists(self.db_name):
            raise FileNotFoundError(f"Vector database không tồn tại tại {self.db_name}. Vui lòng chạy build_index.py trước!")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_db = Chroma(
            persist_directory=self.db_name,
            embedding_function=embedding_model,
            collection_name="movies"
        )
        if not vector_db._collection.count():
            raise ValueError("Vector database trống! Vui lòng chạy build_index.py để thêm dữ liệu.")
        print(f"✅ Database có {vector_db._collection.count()} documents")
        llm = ChatOpenAI(
            temperature=0.3,
            model_name=self.MODEL,
            max_tokens=800,
            streaming=True    # giúp phản hồi từng phần (nếu frontend hỗ trợ)
        )

        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        tools = [
            Tool(
                name="movie_database_search",
                description="Tìm kiếm thông tin phim (tóm tắt, diễn viên, đạo diễn, thể loại, năm sản xuất) từ cơ sở dữ liệu phim nội bộ. Luôn dùng công cụ này trước tiên cho các câu hỏi về phim cụ thể.",
                func=self._movie_search_function
            ),
            web_search_tool,
            cinema_search_tool,
            cinema_showtimes_tool
        ]
        tools.extend(tmdb_tools)  
        

        print("✅ Tất cả components đã sẵn sàng!")
        return llm, retriever, tools

    def _create_agent(self):
        print(f"✨ Tạo agent duy nhất cho phiên chat.")
        prompt = ChatPromptTemplate.from_messages([
            self.SYSTEM_PROMPT,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            max_token_limit=1000,
        )
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            agent_type="react-agent",
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )
        self.memory = memory
        return agent_executor

    def get_response(self, message: str):
        if not message.strip():
            return "Vui lòng nhập câu hỏi của bạn! 🎬"
        try:
            # Log lịch sử
            chat_history = self.memory.chat_memory.messages
            print("\n======================[ AGENT INPUT LOG ]======================")
            print(f"CURRENT CHAT HISTORY ({len(chat_history)} messages):")
            for msg in chat_history:
                print(f"  - {type(msg).__name__}: {msg.content}")
            print(f"INPUT: {message}")
            print("=================================================================\n")
            
            response_dict = self.agent_executor.invoke({"input": message})
            return response_dict.get('output', "Xin lỗi, tôi không thể tạo ra câu trả lời.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Lỗi: {str(e)}"

    def clear_conversation(self):
        self.memory.clear()
        print("🗑️ Đã xóa lịch sử trò chuyện.")
        return "🔄 Đã xóa lịch sử trò chuyện!"

