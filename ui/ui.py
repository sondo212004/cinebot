import gradio as gr
import sys
import os

# Thêm thư mục src vào Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_dir)
from chatbot_engine import ChatbotEngine

class ChatbotUI:
    """Giao diện người dùng cho chatbot"""
    
    def __init__(self):
        # Khởi tạo chatbot engine
        self.engine = ChatbotEngine()
        
        # CSS tùy chỉnh
        self.css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .chat-message {
            padding: 10px !important;
            margin: 5px 0 !important;
            border-radius: 10px !important;
        }
        """
        
        # Các ví dụ câu hỏi
        self.examples = [
            ["Gợi ý cho tôi một bộ phim hành động hay"],
            ["Tôi muốn xem phim tình cảm Hàn Quốc"],
            ["Phim kinh dị nào đáng xem nhất?"],
            ["Bộ phim nào của Christopher Nolan hay nhất?"],
            ["Tôi thích phim anime, có gợi ý gì không?"]
        ]
    
    def _respond(self, message, chat_history=None, *args, **kwargs):
        return self.engine.get_response(message)

    
    def create_interface(self):
        """Tạo Gradio ChatInterface"""
        
        # Tạo ChatInterface với các tham số hợp lệ
        demo = gr.ChatInterface(
            fn=self._respond,
            title="🎬 CineBot - Movie Recommendation Chatbot",
            description="""
            Chào mừng bạn đến với CineBot! 🎭
            
            Tôi là chuyên gia tư vấn phim ảnh, có thể giúp bạn:
            • 🔍 Tìm phim theo thể loại, đạo diễn, diễn viên
            • 💡 Gợi ý phim phù hợp với tâm trạng
            • 📊 So sánh và đánh giá các bộ phim
            • 🌍 Tìm phim từ nhiều quốc gia khác nhau
            
            Hãy hỏi tôi bất cứ điều gì về phim! 🍿
            """,
            examples=self.examples,
            theme=gr.themes.Soft(),
            css=self.css,
            chatbot=gr.Chatbot(
                height=500,
                placeholder="Chưa có tin nhắn nào. Hãy bắt đầu cuộc trò chuyện! 🎬",
                show_label=False,
                avatar_images=["👤", "🤖"],
                type="messages"  # Sử dụng type='messages' để tránh cảnh báo
            ),
            textbox=gr.Textbox(
                placeholder="Nhập câu hỏi về phim...",
                container=False,
                scale=7
            ),
            # Các nút được tự động thêm bởi Gradio, không cần chỉ định retry_btn, undo_btn, v.v.
            additional_inputs=[
                gr.Button(value="🗑️ Xóa lịch sử", interactive=True, elem_id="clear_btn")
            ]
        )
        
        return demo
    
    def launch(self, **kwargs):
        """Khởi chạy giao diện"""
        print("🎬 Khởi động CineBot Movie Recommendation Chatbot...")
        
        try:
            # Tạo Gradio interface
            demo = self.create_interface()
            
            print("✅ Chatbot đã sẵn sàng!")
            print("🌐 Đang mở web interface...")
            
            # Launch với cấu hình tối ưu
            demo.launch(**kwargs)
            
        except Exception as e:
            print(f"❌ Lỗi khởi động ứng dụng: {e}")
            raise

def main():
    """Chạy ứng dụng"""
    try:
        # Tạo UI và khởi chạy
        ui = ChatbotUI()
        ui.launch()
        
    except Exception as e:
        print(f"❌ Lỗi khởi động: {e}")
        raise

if __name__ == "__main__":
    main()