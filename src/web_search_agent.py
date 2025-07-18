from langchain.tools import Tool
from langchain_tavily import TavilySearch
import os

# Cấu hình API key
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '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')

print(f"✅ API Key: {TAVILY_API_KEY[:8]}...")

# Tạo instance TavilySearch với tham số đúng
try:
    tavily_search = TavilySearch(
        max_results=5,
        tavily_api_key=TAVILY_API_KEY,
        search_type="web",  # Chọn loại tìm kiếm là web
        search_engine="google",  # Chọn công cụ tìm kiếm là Google
        days=30  # Tìm kiếm trong 30 ngày gần đây
    )
    print("✅ TavilySearch instance đã được tạo thành công")
except Exception as e:
    print(f"❌ Lỗi khi tạo TavilySearch: {e}")
    tavily_search = None

# Hàm wrapper với xử lý lỗi tốt hơn
def tavily_search_func(query: str) -> str:
    if not tavily_search:
        return "❌ TavilySearch chưa được khởi tạo đúng cách"
    
    try:
        print(f"🔍 Đang tìm kiếm: {query}")
        results = tavily_search.run(query)
        print(f"📊 Loại kết quả: {type(results)}")
        
        if not results:
            return "❌ Không tìm thấy kết quả nào."

        # Xử lý kết quả trả về
        if isinstance(results, str):
            return f"📝 Kết quả tìm kiếm:\n{results}"
        
        elif isinstance(results, list):
            output = []
            for idx, item in enumerate(results, 1):
                if isinstance(item, dict):
                    title = item.get("title", "Không có tiêu đề")
                    content = item.get("content", "Không có nội dung")
                    url = item.get("url", "Không có URL")
                    
                    result_text = (
                        f"📄 Kết quả {idx}:\n"
                        f"Tiêu đề: {title}\n"
                        f"Nội dung: {content[:200]}{'...' if len(content) > 200 else ''}\n"
                        f"URL: {url}\n"
                    )
                else:
                    result_text = f"📄 Kết quả {idx}:\n{str(item)}"
                
                output.append(result_text)
            
            return "\n" + "="*50 + "\n".join(output)
        
        else:
            return f"📝 Kết quả: {str(results)}"
            
    except Exception as e:
        return f"❌ Lỗi khi tìm kiếm: {str(e)}\n📋 Loại lỗi: {type(e).__name__}"

# Định nghĩa Tool cho LangChain
web_search_tool = Tool(
    name="TavilyWebSearch", 
    func=tavily_search_func,
    description=(
        "Dùng để tìm kiếm thông tin trên web qua Tavily API. "
        "Trả về tiêu đề, nội dung, URL của các kết quả tìm kiếm."
    )
)


