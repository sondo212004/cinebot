import os
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import shutil

# Import mới để tránh deprecation warning
try:
    from langchain_chroma import Chroma
    print("✅ Sử dụng langchain_chroma mới")
except ImportError:
    from langchain.vectorstores import Chroma
    print("⚠️ Sử dụng Chroma cũ")

db_name = 'vector_db'

def build_movie_database():
    """Xây dựng vector database từ file JSON"""
    
    print("🎬 Bắt đầu xây dựng Movie Vector Database...")
    
    # Xóa database cũ nếu tồn tại
    if os.path.exists(db_name):
        print(f"🗑️ Xóa database cũ: {db_name}")
        shutil.rmtree(db_name)
    
    documents = []
    
    # Đường dẫn file
    file_path = "data/movies.json"
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        print("💡 Vui lòng đảm bảo file movies.json tồn tại trong thư mục data/")
        return False
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📁 Đã load {len(data)} phim từ {file_path}")
    except Exception as e:
        print(f"❌ Lỗi đọc file JSON: {e}")
        return False
    
    # Xử lý từng phim
    for i, movie in enumerate(data):
        try:
            # Xử lý Title - có thể là string hoặc list
            title = movie.get('Title', 'Unknown')
            if isinstance(title, list):
                title = ', '.join(title)
            
            # Tạo nội dung document
            content = f"""Tên phim: {title}
                        Năm phát hành: {movie.get('Release_year', movie.get('Release Year', 'Không rõ'))}
                        Quốc gia: {movie.get('Nation', movie.get('Origin/Ethnicity', 'Không rõ'))}
                        Đạo diễn: {movie.get('Director', 'Không rõ')}
                        Diễn viên: {movie.get('Cast', 'Không rõ')}
                        Thể loại: {movie.get('Genre', 'Không rõ')}
                        Cốt truyện: {movie.get('Plot', 'Không có mô tả')}
                        Wiki: {movie.get('Wiki_page', movie.get('Wiki Page', 'Không có'))}"""
            
            # Tạo metadata chuẩn
            metadata = {
                "title": title,
                "release_year": str(movie.get('Release_year', movie.get('Release Year', 'Unknown'))),
                "nation": movie.get('Nation', movie.get('Origin/Ethnicity', 'Unknown')),
                "director": movie.get('Director', 'Unknown'),
                "cast": movie.get('Cast', 'Unknown'),
                "genre": movie.get('Genre', 'Unknown'),
                "wiki_page": movie.get('Wiki_page', movie.get('Wiki Page', 'Unknown')),
                "plot": movie.get('Plot', 'No plot available'),
                "source": f"movies.json - {title}"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
            
        except Exception as e:
            print(f"⚠️ Lỗi xử lý phim {i}: {e}")
            continue
    
    print(f"📄 Đã tạo {len(documents)} documents")
    
    if not documents:
        print("❌ Không có document nào được tạo!")
        return False
    
    # Tạo text splitter
    print("✂️ Đang chia nhỏ documents...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📝 Đã tạo {len(chunks)} chunks")
    
    # Tạo embeddings
    print("🤖 Đang tạo embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model đã sẵn sàng")
    except Exception as e:
        print(f"❌ Lỗi tạo embedding model: {e}")
        return False
    
    # Tạo vector store
    print("💾 Đang tạo vector database...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_name,
            collection_name="movies"
        )
        print(f"✅ Đã tạo vector database tại {db_name}")
        
        # Test database
        test_results = vectorstore.similarity_search("phim hành động", k=2)
        print(f"🧪 Test search: Tìm thấy {len(test_results)} kết quả")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo vector database: {e}")
        return False

def test_database():
    """Test vector database"""
    print("\n🧪 Testing Vector Database...")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings,
            collection_name="movies"
        )
        
        # Test queries
        test_queries = [
            "phim hành động",
            "phim tình cảm",
            "phim kinh dị",
            "Christopher Nolan"
        ]
        
        for query in test_queries:
            results = vectorstore.similarity_search(query, k=2)
            print(f"Query: '{query}' -> {len(results)} kết quả")
            if results:
                print(f"  Phim đầu tiên: {results[0].metadata.get('title', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test database: {e}")
        return False

if __name__ == "__main__":
    success = build_movie_database()
    if success:
        print("\n🎉 Xây dựng database thành công!")
        test_database()
        print("\n💡 Bây giờ bạn có thể chạy chatbot_engine.py")
    else:
        print("\n❌ Xây dựng database thất bại!")