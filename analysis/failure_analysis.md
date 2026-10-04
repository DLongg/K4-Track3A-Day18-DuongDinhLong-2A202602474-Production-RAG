# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Dương Đình Long  
**Khóa:** K4 - Track 3A  
**MSSV:** 2A202602474  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Đánh giá |
|--------|:--------------:|:----------:|:---:|:---|
| **Faithfulness** | 0.7949 | 0.5648 | -0.2301 | Bị ảnh hưởng bởi một số câu trả lời vắn tắt hoặc xung đột phiên bản |
| **Answer Relevancy** | 0.7170 | 0.6604 | -0.0565 | Trả lời đúng trọng tâm nhưng một số câu chưa diễn giải đủ chi tiết |
| **Context Precision** | 0.9697 | 0.7917 | -0.1780 | Đạt chuẩn threshold (≥ 0.75), Reranker đẩy đúng chunk liên quan lên top |
| **Context Recall** | 0.9000 | 0.8333 | -0.0667 | Vượt ngưỡng 0.75, độ phủ ngữ cảnh cao qua Hybrid Search + Enrichment |

---

## Bottom-5 Failures

### #1
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Không tìm thấy.
- **Worst metric:** Context Recall (0.00) / Faithfulness (0.00)
- **Error Tree:** Output sai ("Không tìm thấy") → Context đúng? KHÔNG. Context trả về thiếu chunk chứa bảng thâm niên chính sách 2024 → Query OK? Câu hỏi dùng từ "thâm niên" nhưng trong văn bản nội dung phân tán giữa bảng tính thâm niên và bảng lương.
- **Root cause:** Trong bước Hierarchical chunking, child size 256 ký tự đã cắt ngang bảng quy định thâm niên khiến ngữ cảnh bị tách rời; đồng thời Cross-Encoder Top 3 bị chiếm bởi các chunk chính sách nghỉ phép chung.
- **Suggested fix:** Tăng kích thước Child chunk lên 384-512 ký tự hoặc áp dụng Structure-Aware chunking cho các file có bảng biểu markdown, đồng thời tăng `RERANK_TOP_K` lên 5.

---

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.00) / Context Recall (0.00)
- **Error Tree:** Output sai ("Không tìm thấy") → Context đúng? KHÔNG. Dữ liệu có cả `mat_khau_v1.md` (90 ngày) và `mat_khau_v2.md` (120 ngày) → Retrieval/Reranker bị nhiễu do xung đột phiên bản.
- **Root cause:** Xung đột tài liệu đa phiên bản (Multi-version Conflict). Cả hai file đều có nội dung tương đồng cao, khiến BM25 và Dense Search phân tán điểm số, không đưa được chunk `v2.0` vào top xếp hạng cuối cùng.
- **Suggested fix:** Bổ sung metadata filtering theo phiên bản (ví dụ: `status: active` hoặc `version: 2.0`), loại bỏ hoặc gán trọng số thấp cho các tài liệu chính sách cũ đã hết hiệu lực.

---

### #3
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên bắt buộc phải kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ.
- **Worst metric:** Context Recall (0.50)
- **Error Tree:** Output đúng ý chính nhưng thiếu đối chiếu lịch sử chính sách → Context đúng? Context chỉ chứa chunk v2.0, thiếu chunk v1.0 → Query OK? Query dạng câu hỏi tổng quát không ghi rõ phiên bản.
- **Root cause:** Pipeline chỉ retrieve được văn bản chính sách mới nhất mà không lấy được văn bản lịch sử để cung cấp thông tin so sánh như trong Ground Truth.
- **Suggested fix:** Cải tiến kỹ thuật Query Expansion / HyDE để tự động mở rộng câu hỏi thành: "Quy định kích hoạt MFA hiện tại và so với chính sách cũ".

---

### #4
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm.
- **Got:** Nhân viên được nghỉ 3 ngày làm việc khi kết hôn.
- **Worst metric:** Faithfulness (do thiếu mệnh đề "có lương" và "không trừ vào phép năm")
- **Error Tree:** Output thiếu điều kiện ràng buộc → Context đúng? CÓ, context từ `nghi_phep_dac_biet.md` có đầy đủ cả câu → Prompt generation quá ngắn gọn khiến LLM cắt bớt chi tiết.
- **Root cause:** System prompt hiện tại chỉ định: *"Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"*, khiến mô hình LLM thiên về việc rút gọn cực đoan, chỉ trả lời đúng số ngày mà bỏ sót điều kiện kèm theo.
- **Suggested fix:** Cải tiến prompt generation: *"Trả lời chính xác số lượng kèm theo đầy đủ các điều kiện (chế độ hưởng lương, thủ tục, có trừ phép năm hay không) được nêu trong context."*

---

### #5
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng.
- **Got:** Không, nhân viên không nên tự xử lý malware trên máy mà không có sự hướng dẫn của đội CNTT.
- **Worst metric:** Answer Relevancy (thiếu SLA 1 giờ và kênh liên hệ cụ thể)
- **Error Tree:** Output đúng phán quyết (Không) nhưng thiếu kênh liên hệ khẩn cấp và thời hạn xử lý → Context đúng? CÓ, chunk có nêu rõ thời hạn 1 giờ và email/hotline → LLM summarize quá mức.
- **Root cause:** Mô hình sinh câu trả lời mang tính hội thoại chung chung thay vì trích xuất chi tiết hành động hành chính bắt buộc.
- **Suggested fix:** Áp dụng Chain-of-Thought hoặc định dạng cấu trúc câu trả lời: gồm (1) Kết luận rõ ràng, (2) Thời hạn hành động (SLA), (3) Kênh liên hệ xử lý khẩn cấp.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"*

**Error Tree walkthrough:**
1. **Output đúng?** → KHÔNG. Mô hình trả lời "Không tìm thấy" trong khi quy định có sẵn trong tập tài liệu.
2. **Context đúng?** → KHÔNG. Context được rerank và truyền vào prompt không chứa đoạn văn bản từ `nghi_phep_nam_v2024.md` nói về mốc 3 năm, mà bị lấn át bởi các quy định chung của `nghi_phep_nam_v2023.md`.
3. **Query rewrite / Hybrid Search OK?** → Truy vấn gốc chỉ có 11 từ tiếng Việt, BM25 match từ "thâm niên" nhưng trong tài liệu v2024 từ này nằm trong câu: *"Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 3 năm thâm niên công tác"*. Bước chunking hierarchical đã chia nhỏ paragraph này ra làm mất ngữ cảnh tiêu đề.
4. **Fix ở bước:**
   - **Bước 1 (Chunking):** Dùng Structure-Aware Chunking để giữ nguyên tiêu đề section cùng nội dung bảng tính thâm niên.
   - **Bước 2 (Enrichment):** Contextual Prepend bổ sung rõ ràng `"Tài liệu: Quy định nghỉ phép năm 2024 (mới nhất)"` vào trước chunk.
   - **Bước 3 (Reranking):** Lấy Top 5 thay vì Top 3 để đảm bảo context đầy đủ cho LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Thứ nhất:** Triển khai **Query Expansion / Query Rewriting** bằng LLM để tự động phân rã các câu hỏi đối chiếu hoặc câu hỏi phức (multi-hop).
- **Thứ hai:** Tích hợp **Metadata Version Filter** vào Qdrant để tự động ưu tiên tài liệu có hiệu lực mới nhất (v2024 > v2023), triệt tiêu hoàn toàn lỗi xung đột phiên bản.
- **Thứ ba:** Tinh chỉnh Prompt Template của khâu Generation với cấu trúc Output bắt buộc rõ ràng, tăng điểm Faithfulness lên $\ge 0.85$.
