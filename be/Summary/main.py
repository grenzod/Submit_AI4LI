import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re

dotenv.load_dotenv()

class SummaryProcessor:
    def __init__(self):
        self.groq = init_chat_model(
            "llama3-70b-8192", 
            model_provider="groq",
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
            frequency_penalty=0.1
        )
        self.tokenizer = self.groq.get_tokenizer()
    
    def clean_and_join(self, text: str) -> str:
        text = re.sub(r'[^\w\u00C0-\u024F\s.,!?;:\'"-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_summary(self, text_chunk: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "Bạn là trợ lý tóm tắt chuyên nghiệp. Hãy tạo bản tóm tắt ngắn gọn, súc tích "
                "cho đoạn văn bản dưới đây, tập trung vào thông tin quan trọng nhất. "
                "Chỉ trả lời bằng bản tóm tắt, không thêm bất kỳ nội dung nào khác."
            )),
            HumanMessage(content=f"### Văn bản cần tóm tắt:\n{text_chunk}")
        ])
        
        chain = prompt | self.groq | StrOutputParser()
        
        try:
            response = chain.invoke({})
            
            if "tóm tắt:" in response.lower():
                parts = re.split(r'tóm tắt:\s*', response, flags=re.IGNORECASE)
                return parts[-1].strip()
            return response.strip()
        
        except Exception as e:
            print(f"Lỗi khi tạo tóm tắt: {e}")
            return "Không thể tạo tóm tắt cho đoạn này"

    def summarize_file(self, file_path: str, max_tokens: int = 1024) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            
            clean = self.clean_and_join(raw)
            
            sentences = re.split(r'(?<=[.!?])\s+', clean)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current = ""
            
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip() if current else sentence
                token_count = len(self.tokenizer.encode(candidate, truncation=False))
                
                if token_count <= max_tokens:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
            
            if current:
                chunks.append(current)
            
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Đang xử lý chunk {i+1}/{len(chunks)} ({len(chunk)} ký tự)")
                summary = self.generate_summary(chunk)
                summaries.append(summary)
            
            return "\n\n".join(summaries)
        
        except Exception as e:
            print(f"Lỗi khi xử lý file: {e}")
            return "Đã xảy ra lỗi trong quá trình tóm tắt"
