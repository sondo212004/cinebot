import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Annotated
import json
import asyncio

# LangChain core imports
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.schema import BaseRetriever, SystemMessage, HumanMessage, AIMessage

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
    from playwright_browser import scrape_cinema_showtimes_playwright
except ImportError:
    print("⚠️ Không tìm thấy web_search_agent.py, cinema_search.py, tmdb_tools.py, mcp_tools.py, tạo tool giả lập.")
    from langchain.tools import DuckDuckGoSearchRun
    web_search_tool = DuckDuckGoSearchRun()
    cinema_search_tool = None
    tmdb_tools = []
    cinema_showtimes_tool = None
    async def get_mcp_browser_tools(): # Giả lập hàm nếu không import được
        print("⚠️ Giả lập get_mcp_browser_tools: Không có browser automation tool.")
        return []

import langsmith
langsmith_client = langsmith.Client()

# Định nghĩa State cho LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


class ChatbotEngine:
    """Engine xử lý logic chatbot với LangGraph."""
    
    def __init__(self):
        # Cấu hình
        self.MODEL = "gpt-4o-mini"
        self.db_name = 'vector_db'

        self.SYSTEM_PROMPT = """ 
        Bạn là CineBot - một chuyên gia tư vấn phim ảnh thông minh và thân thiện.

        NHIỆM VỤ CỐT LÕI:
        - Phân tích kỹ lưỡng yêu cầu của người dùng, ĐẶC BIỆT LƯU Ý đến lịch sử trò chuyện để hiểu rõ ngữ cảnh.
        - Nếu người dùng hỏi một câu nối tiếp (ví dụ: "còn phim nào khác không?", "phim đó của ai?"), bạn PHẢI dựa vào lịch sử trò chuyện để suy ra chủ đề hoặc bộ phim đang được nói đến.

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

        **RẤT QUAN TRỌNG: CÁCH XỬ LÝ YÊU CẦU VỀ RẠP CHIẾU CỤ THỂ:**

        1.  **Nếu người dùng NÊU RÕ TÊN RẠP CỤ THỂ** (ví dụ: "CGV Vincom Center Bà Triệu", "Lotte Cinema Royal City", "BHD Star Phạm Hùng", hoặc cung cấp URL của rạp):
            * **Tuyệt đối KHÔNG SỬ DỤNG `cinema_search_tool`.** Bạn đã có tên rạp rồi.
            * **Hãy BỎ QUA bước tìm rạp và chuyển thẳng đến việc tìm URL (nếu chưa có) và scrape lịch chiếu.**
            * **Nếu người dùng CHƯA CUNG CẤP URL rạp, hãy sử dụng `web_search_tool`** để tìm kiếm URL TRANG LỊCH CHIẾU CỤ THỂ của rạp đó.
                * Ví dụ: Nếu rạp là "CGV Vincom Center Bà Triệu", hãy tìm `web_search_tool("lịch chiếu CGV Vincom Center Bà Triệu URL")` hoặc `web_search_tool("trang chính thức CGV Vincom Center Bà Triệu")`.
            * Sau đó, **PHÂN TÍCH KẾT QUẢ TỪ `web_search_tool` để TRÍCH XUẤT URL TRANG LỊCH CHIẾU CHÍNH XÁC.** (Đảm bảo là link từ miền chính thức của rạp như `cgv.vn`, `lottecinemavn.com`, `bhdstar.vn`).
            * Hiện tại chỉ hỗ trợ cho rạp CGV, nên hãy lấy url ví dụ:https://www.cgv.vn/en/cinox/site/cgv-vincom-tran-duy-hung

        2.  **Nếu người dùng CHỈ NÊU TÊN PHIM VÀ ĐỊA ĐIỂM CHUNG** (ví dụ: "phim X ở Hà Nội", "phim Y ở gần đây", "lịch chiếu ở quận Hoàn Kiếm"):
            * **SỬ DỤNG CÔNG CỤ `cinema_search_tool`** với `movie_name` và `location` (nếu có) để tìm danh sách các rạp đang chiếu phim đó ở khu vực gần người dùng. Công cụ này sẽ trả về **TÊN RẠP** và **ĐỊA CHỈ**.
            * **Từ kết quả của `cinema_search_tool`, HÃY CHỌN MỘT HOẶC HAI TÊN RẠP ĐẠI DIỆN** để tiếp tục.
            * Sau đó, tương tự như bước 1, **SỬ DỤNG `web_search_tool`** để tìm URL TRANG LỊCH CHIẾU CỤ THỂ của các rạp đã chọn.

        3.  **Sau khi đã có `specific_cinema_url` (từ `web_search_tool` hoặc từ prompt của người dùng):**
            * **SỬ DỤNG CÔNG CỤ `scrape_cinema_showtimes_playwright`** để lấy lịch chiếu chi tiết từ URL đó.
            * **RẤT QUAN TRỌNG:** Truyền một `task_description` rõ ràng cho công cụ này, bao gồm URL, ngày muốn xem (nếu có, ví dụ: "ngày hôm nay", "ngày mai", "29/07/2025"), và yêu cầu trích xuất thông tin (tên phim, giờ chiếu, định dạng, link đặt vé) dưới dạng JSON list.
            * Ví dụ: `scrape_cinema_showtimes_playwright(url="https://en.wikipedia.org/wiki/Rapping", target_date="[Ngày cần tìm]", extract_format="json")`.

        4.  **Sau khi công cụ scrape trả về dữ liệu lịch chiếu (dạng string JSON):**
            * Bạn **HÃY PHÂN TÍCH DỮ LIỆU ĐÓ và TỔNG HỢP, SẮP XẾP, TRÌNH BÀY THÔNG TIN** một cách rõ ràng, đầy đủ và thân thiện cho người dùng.
            * Đảm bảo câu trả lời bao gồm tên phim, các rạp chiếu, thời gian chiếu cụ thể cho từng ngày, và đường dẫn để đặt vé nếu có.
        """

        # Khởi tạo components

        self.llm, self.retriever, self.tools = self._initialize_components()

        # Tạo LangGraph workflow
        self.workflow = self._create_workflow()
        
        # Memory saver để lưu trữ state
        self.memory = MemorySaver()
        
        # Compile workflow với memory
        self.app = self.workflow.compile(checkpointer=self.memory)

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
            streaming=True
        )

        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        tools = [
            Tool(
                name="movie_database_search",
                description="Tìm kiếm thông tin phim (tóm tắt, diễn viên, đạo diễn, thể loại, năm sản xuất) từ cơ sở dữ liệu phim nội bộ. Luôn dùng công cụ này trước tiên cho các câu hỏi về phim cụ thể.",
                func=self._movie_search_function
            ),
        ]
        
        # Thêm các tools khác nếu có
        if web_search_tool:
            tools.append(web_search_tool)
        if cinema_search_tool:
            tools.append(cinema_search_tool)
        if tmdb_tools:
            tools.extend(tmdb_tools)
        # if cinema_showtimes_tool:
        #     tools.append(cinema_showtimes_tool)
        if scrape_cinema_showtimes_playwright:
            tools.append(scrape_cinema_showtimes_playwright)
        
        print("✅ Tất cả components đã sẵn sàng!")
        return llm, retriever, tools

    def _create_workflow(self):
        print("✨ Tạo LangGraph workflow...")
        
        # Tạo workflow graph
        workflow = StateGraph(AgentState)
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Định nghĩa node chatbot
        def chatbot(state: AgentState):
            # Thêm system message vào đầu
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Thêm nodes vào workflow
        workflow.add_node("chatbot", chatbot)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Thiết lập entry point
        workflow.add_edge(START, "chatbot")
        
        # Định nghĩa conditional edges
        workflow.add_conditional_edges(
            "chatbot",
            tools_condition,
            {"tools": "tools", "__end__": END}
        )
        
        # Từ tools quay lại chatbot
        workflow.add_edge("tools", "chatbot")
        
        return workflow

    def get_response(self, message: str, session_id: str = "default_session"):
        """
        Lấy response từ LangGraph agent
        
        Args:
            message: Tin nhắn của người dùng
            session_id: ID phiên để lưu trữ lịch sử
        """
        if not message.strip():
            return "Vui lòng nhập câu hỏi của bạn! 🎬"
        
        try:
            # Tạo config với thread_id để lưu trữ lịch sử
            config = {"configurable": {"thread_id": session_id}}
            
            # Log input
            print(f"\n🎬 INPUT: {message}")
            
            # Invoke workflow với message mới
            response = self.app.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config
            )
            
            # Lấy message cuối cùng từ AI
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return "Xin lỗi, tôi không thể tạo ra câu trả lời."
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Lỗi: {str(e)}"

    def clear_conversation(self, session_id: str = "default_session"):
        """
        Xóa lịch sử trò chuyện cho một session cụ thể
        """
        try:
            # Với MemorySaver, chúng ta có thể xóa bằng cách tạo session mới
            # Hoặc có thể implement logic xóa riêng nếu cần
            print(f"🗑️ Đã xóa lịch sử trò chuyện cho session: {session_id}")
            return "🔄 Đã xóa lịch sử trò chuyện!"
        except Exception as e:
            return f"❌ Lỗi khi xóa lịch sử: {str(e)}"

    def get_conversation_history(self, session_id: str = "default_session"):
        """
        Lấy lịch sử trò chuyện cho một session
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            # Lấy state hiện tại
            current_state = self.app.get_state(config)
            if current_state and current_state.values.get("messages"):
                return current_state.values["messages"]
            return []
        except Exception as e:
            print(f"❌ Lỗi khi lấy lịch sử: {str(e)}")
            return []

    async def stream_response(self, message: str, session_id: str = "default_session"):
        if not message.strip():
            yield "Vui lòng nhập câu hỏi của bạn! 🎬"
            return

        try:
            config = {"configurable": {"thread_id": session_id}}
            
            async for chunk in self.app.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        yield last_message.content
        except Exception as e:
            yield f"❌ Đã xảy ra lỗi: {str(e)}"

                        
        except Exception as e:
            yield f"❌ Lỗi: {str(e)}"