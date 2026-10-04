# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Dương Đình Long  
**Khóa:** K4 - Track 3A  
**MSSV:** 2A202602474  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu toàn bộ các khái niệm cốt lõi từ bài giảng lý thuyết vào việc hiện thực hóa mã nguồn trong Lab 18:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích kỹ thuật |
|---|:---:|---|---|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Sử dụng mô hình `all-MiniLM-L6-v2` tính Cosine Similarity giữa các câu liền kề với ngưỡng `threshold=0.85`. Nhóm các câu có cùng trường nghĩa vào chung một chunk. Tránh hoàn toàn tình trạng cắt ngang mạch lập luận như phương pháp cắt đoạn cố định (fixed-size). |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Tạo cấu trúc phân tầng: Parent Chunk (2048 ký tự) chứa ngữ cảnh tổng quan của tài liệu, Child Chunk (256 ký tự) chứa chi tiết cụ thể kèm `parent_id`. Khi truy vấn, hệ thống tìm kiếm trên tập Child để đạt độ chính xác cao (Precision), sau đó trả về Parent để cung cấp ngữ cảnh đầy đủ cho LLM. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Sử dụng biểu thức chính quy (Regex) phân tích các cấp tiêu đề Markdown (`#`, `##`, `###`), bảo toàn nguyên vẹn cấu trúc bảng biểu, code block và danh sách. Gán metadata `section` giúp phục vụ metadata filtering ở bước sau. |
| **Vietnamese Word Segmentation & BM25** | M2 | `segment_vietnamese()` & `BM25Search` | Áp dụng `underthesea.word_tokenize` và xử lý chuyển đổi `_` thành khoảng trắng `" "` nhằm đồng bộ token space với thuật toán `BM25Okapi`. Khắc phục triệt để hiện tượng trượt từ khóa đặc thù tiếng Việt khi người dùng tìm kiếm từ ghép. |
| **Dense Vector & Qdrant Search** | M2 | `DenseSearch` | Tích hợp mô hình đa ngôn ngữ `BAAI/bge-m3` (dimension 1024) và hệ cơ sở dữ liệu vector Qdrant. Triển khai cơ chế tự động fallback sang `:memory:` khi chưa khởi chạy container Docker, đảm bảo hệ thống vận hành liên tục không gián đoạn. |
| **Reciprocal Rank Fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | Áp dụng công thức $RRF\_Score(d) = \sum \frac{1}{60 + rank + 1}$ để dung hợp bảng xếp hạng từ BM25 (lexical) và Dense Search (semantic). Giúp các tài liệu vừa khớp từ khóa vừa mang ngữ nghĩa tương đồng được đẩy lên vị trí dẫn đầu. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình `BAAI/bge-reranker-v2-m3` tính toán full-attention đồng thời trên cặp `(query, document)`. Tinh lọc từ Top 20 ứng viên thu về Top 3 context cô đọng và liên quan nhất, loại bỏ tối đa nhiễu ngữ cảnh (context noise). |
| **RAGAS 4 Core Metrics** | M4 | `evaluate_ragas()` | Đánh giá tự động toàn diện qua 4 chỉ số: Faithfulness (độ trung thực), Answer Relevancy (độ liên quan của câu trả lời), Context Precision (độ chuẩn xác xếp hạng context), Context Recall (độ bao phủ thông tin gốc). Đạt kết quả định lượng phục vụ so sánh giữa Baseline và Production. |
| **Diagnostic Tree Failure Analysis** | M4 | `failure_analysis()` | Xây dựng cây chẩn đoán tự động phân loại lỗi: nhận diện nguyên nhân gốc rễ (LLM hallucinating, Missing chunks, Irrelevant chunks, Prompt misalignment) và tự động đưa ra giải pháp khắc phục có tính hệ thống. |
| **Contextual Enrichment (Single-Call Mode)** | M5 | `_enrich_single_call()` & `enrich_chunks()` | Triển khai kỹ thuật Contextual Retrieval của Anthropic kết hợp HyQA và Metadata Extraction. Tối ưu chi phí bằng phương pháp gộp 4 tác vụ vào 1 lần gọi API duy nhất (Structured JSON output), tăng tốc độ xử lý bằng ThreadPoolExecutor song song. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi thiếu thư viện `rank-bm25`
- **Exact Error Message:**
  ```text
  ModuleNotFoundError: No module named 'rank_bm25'
  ```
- **Nguyên nhân & Cách debug:**
  Môi trường Python cục bộ ban đầu chưa cài đặt gói `rank-bm25`. Khi chạy unit test `tests/test_m2.py`, quá trình import trong `BM25Search.index()` bị ném ngoại lệ.
  *Khắc phục:* Chạy `pip install rank-bm25` và đồng bộ toàn bộ gói phụ thuộc thông qua `pip install -r requirements.txt`.

### 2. Lỗi token mismatch trong tiếng Việt giữa Underthesea và BM25
- **Hiện tượng lỗi:**
  Hàm `underthesea.word_tokenize(text, format="text")` tự động nối các từ ghép bằng dấu gạch dưới (ví dụ: `"nghỉ_phép"`). Trong khi đó, thuật toán `BM25Okapi` phân tách token theo khoảng trắng đơn lẻ. Khi người dùng nhập truy vấn `"nghỉ phép"`, truy vấn sinh ra 2 tokens riêng biệt `["nghỉ", "phép"]`, dẫn đến việc không khớp với token `"nghỉ_phép"` trong kho văn bản đã index.
- **Cách khắc phục:**
  Trong hàm `segment_vietnamese()`, thực hiện chuẩn hóa:
  ```python
  segmented = word_tokenize(text, format="text")
  return segmented.replace("_", " ")
  ```
  Nhờ đó, BM25 so khớp từ vựng chính xác 100% trên các câu truy vấn tiếng Việt.

### 3. Khó khăn khi tải và nạp mô hình Reranker dung lượng lớn trên Windows
- **Hiện tượng:**
  Mô hình `BAAI/bge-reranker-v2-m3` có kích thước file `model.safetensors` lên tới 2.27 GB. Quá trình tải từ HuggingFace Hub qua đường truyền quốc tế mất nhiều thời gian, kèm cảnh báo symlink trên hệ điều hành Windows (`UserWarning: huggingface_hub cache-system uses symlinks by default...`). Ngoài ra, do khởi tạo mới instance trong mỗi test case, mô hình bị reload nhiều lần gây chậm test suite.
- **Cách khắc phục:**
  1. Kiên nhẫn chờ tiến trình hoàn tất tải và xác thực tính toàn vẹn của blob cache trong `$env:USERPROFILE\.cache\huggingface\hub`.
  2. Áp dụng kỹ thuật Module-level Dictionary Caching (`_cross_encoder_cache = {}`) trong `src/m3_rerank.py`, giúp mô hình chỉ nạp vào bộ nhớ RAM duy nhất 1 lần, các lần gọi tiếp theo đạt tốc độ truy xuất tức thì (0ms).

### 4. Xử lý bài toán giới hạn tần suất (Rate Limit / In-Flight Credits) khi chạy RAGAS
- **Exact Error Message:**
  ```text
  APIStatusError: Error code: 402 - {'error': {'message': 'This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle...'}}
  ```
- **Nguyên nhân & Cách debug:**
  RAGAS mặc định sinh ra đồng thời nhiều asynchronous jobs (tính toán song song 4 metrics trên 20 câu hỏi = 80 requests cùng lúc), làm cạn ngân sách in-flight requests của gateway API.
  *Khắc phục:* Thiết lập tham số `raise_exceptions=False` trong hàm `ragas.evaluate()`, bổ sung khối `try/except` an toàn trong `src/m4_eval.py` và tối ưu `enrich_chunks()` với worker pool vừa phải (`max_workers=10`) để cân bằng giữa tốc độ và giới hạn API.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý AI Pháp chế & Tra cứu Quy chế Doanh nghiệp Nội bộ (Corporate Policy & Legal Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Hệ thống đang sử dụng kiến trúc Naive RAG cơ bản: file PDF/Markdown được cắt cố định 500 ký tự với overlap 50 ký tự, sau đó lưu vào ChromaDB và tìm kiếm Dense vector thông qua mô hình embedding mã nguồn mở.
- **Vấn đề / Bottlenecks đang gặp:**
  1. *Xung đột phiên bản văn bản:* Quy chế năm 2024 thay thế quy chế năm 2023 nhưng hệ thống vẫn trả về cả hai, dẫn đến câu trả lời sai lệch thông tin mới nhất.
  2. *Độ trôi ngữ cảnh bảng biểu:* Các điều khoản chứa bảng thang lương, ngày nghỉ theo thâm niên thường bị cắt vụn qua nhiều chunks, khiến LLM trả lời "không tìm thấy" hoặc tính toán sai số ngày phép.
  3. *Hallucination và thiếu căn cứ pháp lý:* Chưa có thước đo định lượng (evaluation metrics) để kiểm soát chất lượng câu trả lời trước khi chuyển đến người dùng cuối.

#### 2. Kế hoạch cải tiến kỹ thuật
1. **Chunking Strategy:**
   - Chuyển sang kết hợp **Structure-Aware Chunking** (nhận diện theo Điều, Khoản, Mục của văn bản quy phạm) và **Hierarchical Chunking** (Parent 2048 ký tự - Child 384 ký tự).
   - Đảm bảo toàn vẹn cấu trúc bảng biểu và duy trì ngữ cảnh toàn diện của văn bản quy chế.
2. **Search Retrieval:**
   - Xây dựng kiến trúc **Hybrid Search**: kết hợp BM25 (được tối ưu tách từ tiếng Việt qua Underthesea) và Dense Vector (mô hình `BAAI/bge-m3`).
   - Hợp nhất kết quả bằng **Reciprocal Rank Fusion (RRF)** với $k=60$, Top 20 candidates.
   - Thêm bộ lọc Metadata bắt buộc: lọc theo `status: active` và `version: latest` để giải quyết triệt để vấn đề xung đột tài liệu cũ/mới.
3. **Reranking:**
   - Triển khai **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`) để tinh lọc từ Top 20 xuống Top 3-5 ngữ cảnh liên quan nhất.
   - Nghiên cứu áp dụng thêm `FlashRank` cho các nghiệp vụ yêu cầu phản hồi siêu tốc (<50ms).
4. **Enrichment:**
   - Áp dụng kỹ thuật **Contextual Prepend** (Anthropic style) thông qua cơ chế Combined Single-Call: tiền xử lý bổ sung ngữ cảnh cấp cao (`Tên văn bản + Số hiệu + Ngày ban hành + Chủ đề điều khoản`) vào đầu mỗi chunk.
5. **Evaluation & CI/CD Pipeline:**
   - Tích hợp bộ đánh giá **RAGAS 4 Core Metrics** vào quy trình CI/CD kiểm thử tự động mỗi khi có văn bản quy chế mới được cập nhật vào kho tri thức.
   - Đặt ngưỡng chặn (Quality Gate): Faithfulness $\ge 0.85$, Context Precision $\ge 0.80$.

#### 3. Timeline triển khai (4 tuần)

| Giai đoạn | Thời gian | Nhiệm vụ trọng tâm | Mục tiêu đầu ra (Deliverables) |
|---|:---:|---|---|
| **Tuần 1** | Ngày 1 – 7 | Tái cấu trúc khâu tiền xử lý dữ liệu: Triển khai Structure-Aware & Hierarchical Chunking cho toàn bộ văn bản quy chế nội bộ. | 100% tài liệu được trích xuất bảo toàn bảng biểu và gán metadata phân tầng. |
| **Tuần 2** | Ngày 8 – 14 | Xây dựng Hybrid Search (BM25 + Qdrant Dense Vector + RRF) và tích hợp Cross-Encoder Reranker. | Pipeline Hybrid Search hoạt động ổn định, Top-3 context đạt độ chính xác cao. |
| **Tuần 3** | Ngày 15 – 21 | Triển khai Combined Enrichment Pipeline (bổ sung Contextual Prepend & Auto Metadata) và tinh chỉnh Generation Prompt. | Hoàn thiện pipeline end-to-end, giảm thiểu tối đa hiện tượng ảo giác (hallucination). |
| **Tuần 4** | Ngày 22 – 28 | Thiết lập bộ Benchmark Test Set gồm 100 câu hỏi nghiệp vụ và tự động hóa đánh giá định lượng bằng RAGAS. | Báo cáo đánh giá RAGAS đạt Faithfulness $\ge 0.85$, sẵn sàng đưa vào môi trường Production. |
