# KẾ HOẠCH HOÀN THÀNH LAB DAY 22 — 100 ĐIỂM

## Tổng quan

Lab yêu cầu hoàn thành 4 task, tổng 100 điểm + tối đa 10 điểm bonus. Kế hoạch này chia nhỏ từng task thành các bước cụ thể để triển khai.

---

## CẤU TRÚC FILE CẦN TẠO

```
├── .env                          # API keys (đã có template)
├── .gitignore                    # Phải exclude .env, __pycache__, *.pyc
├── requirements.txt              # Đã có danh sách packages
├── config.py                     # Load .env, validate keys, in trạng thái
├── qa_pairs.py                   # 50 cặp QA với ground-truth answers
├── data/
│   ├── knowledge_base.txt        # Knowledge base cho RAG
│   └── ragas_report.json         # Sinh ra bởi Step 3
├── evidence/                     # Thư mục submit screenshot + logs
│   ├── 01_langsmith_traces.png
│   ├── 02_prompt_hub.png
│   ├── 02_ab_routing_log.txt
│   ├── 03_ragas_scores.png
│   ├── 03_ragas_report.json
│   ├── 04_pii_demo_log.txt
│   ├── 04_json_demo_log.txt
│   └── README.md                 # Phân tích ngắn gọn kết quả
├── 01_langsmith_rag_pipeline.py  # Step 1
├── 02_prompt_hub_ab_routing.py   # Step 2
├── 03_ragas_evaluation.py        # Step 3
├── 04_guardrails_validator.py    # Step 4
└── run_all.py                    # Chạy tất cả steps
```

---

## CHUẨN BỊ — CƠ SỞ HẠ TẦNG

### 0.1 — Tạo file `.env`
```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=day22-rag-pipeline
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 0.2 — Tạo file `.gitignore`
```
.env
__pycache__/
*.pyc
.DS_Store
```

### 0.3 — Tạo file `config.py`
- Dùng `python-dotenv` load `.env`
- Validate tồn tại các biến môi trường bắt buộc
- In ra trạng thái: `✅ Config loaded successfully` với tên project, endpoint, model

### 0.4 — Tạo file `qa_pairs.py`
- Chứa list `QA_PAIRS` với 50 phần tử, mỗi phần tử có `question` và `reference`
- Copy từ pseudocode `03_ragas_evaluation.py` (đã có sẵn 50 câu)

### 0.5 — Tạo file `data/knowledge_base.txt`
- Viết nội dung kiến thức bao phủ tất cả 50 câu hỏi (ML, DL, NLP, Transformers, RAG, LangChain, LangSmith, RAGAS, Guardrails, AI Safety)
- Độ dài đủ để split thành chunks và retrieval hiệu quả
- Cấu trúc: chia theo chủ đề, mỗi chủ đề 1-2 đoạn

### 0.6 — Cài đặt dependencies
```bash
pip install -r requirements.txt
python config.py  # verify
```

---

## TASK 1 — LANGSMITH RAG PIPELINE (25 điểm)

### Mục tiêu
- Build RAG pipeline với FAISS vector search
- 50 traces xuất hiện trong LangSmith UI

### Các bước triển khai (`01_langsmith_rag_pipeline.py`)

| # | Bước | Điểm |
|---|------|------|
| 1.1 | Load `.env`, set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` **trước khi import LangChain** | — |
| 1.2 | Khởi tạo `ChatOpenAI` với `model=`, `api_key=`, `base_url=` từ biến môi trường | — |
| 1.3 | Khởi tạo `OpenAIEmbeddings` với `model=`, `api_key=`, `base_url=` | — |
| 1.4 | Implement `build_vectorstore()`: đọc `data/knowledge_base.txt` → `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` → `FAISS.from_texts(chunks, embeddings)` | 5 |
| 1.5 | Định nghĩa `RAG_PROMPT = ChatPromptTemplate.from_messages([...])` với system message yêu cầu chỉ dùng context + human `{question}` | — |
| 1.6 | Implement `build_rag_chain(vectorstore)`: retriever (k=3) → format_docs → prompt → llm → StrOutputParser | 5 |
| 1.7 | Implement `ask(chain, question)` với decorator `@traceable(name="rag-query")` gọi `chain.invoke(question)` | 10 |
| 1.8 | Loop qua 50 câu hỏi, gọi `ask()`, in kết quả kèm số thứ tự | 5 |

**Xác minh:** Mở https://smith.langchain.com → thấy ≥ 50 traces

---

## TASK 2 — PROMPT HUB & A/B ROUTING (25 điểm)

### Mục tiêu
- Push 2 prompt versions lên LangSmith Prompt Hub
- Pull về và dùng để routing
- 50 traces bổ sung trong LangSmith (tổng ≥ 100)

### Các bước triển khai (`02_prompt_hub_ab_routing.py`)

| # | Bước | Điểm |
|---|------|------|
| 2.1 | Định nghĩa `PROMPT_V1`: system message yêu cầu trả lời ngắn gọn 2-4 câu, chỉ dùng context | 5 |
| 2.2 | Định nghĩa `PROMPT_V2`: system message yêu cầu trả lời chi tiết 3-5 câu, có cấu trúc rõ ràng | 5 |
| 2.3 | Implement `push_prompts_to_hub(client)`: gọi `client.push_prompt(name, object=template)` cho cả V1 và V2 | 8 |
| 2.4 | Implement `pull_prompts_from_hub(client)`: gọi `client.pull_prompt(name)`, có fallback về local nếu lỗi | 4 |
| 2.5 | Implement `get_prompt_version(request_id)`: `hashlib.md5(request_id) → int → % 2 → V1 (chẵn) / V2 (lẻ)` | 5 |
| 2.6 | Implement `ask_ab(retriever, llm, prompt, question, version)`: retrieve docs → format context → invoke chain → return dict | — |
| 2.7 | Main loop: với mỗi câu hỏi, tạo `request_id`, route đến version tương ứng, log `[prompt-v1]` hoặc `[prompt-v2]` | 3 |
| 2.8 | In tổng kết: bao nhiêu câu routed đến V1, bao nhiêu đến V2 | — |

**Xác minh:**
- Prompt Hub UI hiển thị 2 prompt versions
- Console output hiển thị mix `[prompt-v1]` và `[prompt-v2]`
- LangSmith có thêm 50 traces (tổng ≥ 100)

---

## TASK 3 — RAGAS EVALUATION (25 điểm)

### Mục tiêu
- Đánh giá cả 2 prompt versions với 4 RAGAS metrics
- Faithfulness ≥ 0.8 cho ít nhất 1 version
- Lưu kết quả ra `data/ragas_report.json`

### Các bước triển khai (`03_ragas_evaluation.py`)

| # | Bước | Điểm |
|---|------|------|
| 3.1 | Import RAGAS: `from ragas import evaluate, EvaluationDataset, SingleTurnSample` | — |
| 3.2 | Import 4 metrics: `from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision` | — |
| 3.3 | Implement `run_rag(retriever, llm, prompt, question) → {"answer": str, "contexts": list[str]}` | — |
| 3.4 | Implement `collect_rag_outputs(vectorstore, prompt_version)`: chạy 50 QA pairs qua 1 prompt version, thu thập results | 5 |
| 3.5 | Implement `build_ragas_dataset(rag_results)`: tạo `SingleTurnSample` cho từng kết quả, build `EvaluationDataset` | 5 |
| 3.6 | Implement `run_ragas_eval(rag_results, version)`: tạo dataset → evaluate với 4 metrics → tính mean scores | 8 |
| 3.7 | Main: build vectorstore → collect outputs cho V1 và V2 → evaluate cả hai | — |
| 3.8 | In bảng so sánh: từng metric với V1 score, V2 score, và winner indicator | — |
| 3.9 | Kiểm tra faithfulness ≥ 0.8 và in `✅ Target met` hoặc `⚠️ Below target` | 5 |
| 3.10 | Lưu `data/ragas_report.json` với `prompt_v1_scores`, `prompt_v2_scores`, `target_met` | 2 |

**Lưu ý:**
- `SingleTurnSample.retrieved_contexts` phải là `list[str]`, không được join
- `result[metric]` trả về list floats → dùng `numpy.mean()` để tính trung bình
- Step này chạy ~20-30 phút do RAGAS gọi LLM nhiều lần
- Thêm `warnings.filterwarnings("ignore")` để suppress deprecation warnings

**Xác minh:**
- Bảng so sánh V1 vs V2 hiển thị đủ 4 metrics
- Ít nhất 1 version có faithfulness ≥ 0.8
- File `data/ragas_report.json` tồn tại

---

## TASK 4 — GUARDRAILS AI VALIDATORS (25 điểm)

### Mục tiêu
- 2 custom validators: PII Detector + JSON Formatter
- Demo với test cases

### Validator A — PII Detector (13 điểm)

| # | Bước | Điểm |
|---|------|------|
| 4.1 | Tạo class `PIIDetector(Validator)` với `@register_validator(name="pii-detector", data_type="string")` | 3 |
| 4.2 | Định nghĩa regex patterns cho: EMAIL, PHONE (US), SSN, CREDIT_CARD | 5 |
| 4.3 | Implement `validate(value, metadata)`: duyệt từng pattern → nếu tìm thấy → replace bằng `[TYPE_REDACTED]` → `PassResult(value_override=redacted_text)` | 3 |
| 4.4 | Test với 6 test cases: Email, Phone, SSN, Credit Card, Multi-PII, Clean | 2 |

### Validator B — JSON Formatter (12 điểm)

| # | Bước | Điểm |
|---|------|------|
| 4.5 | Tạo class `JSONFormatter(Validator)` với `@register_validator(name="json-formatter", data_type="string")` | 3 |
| 4.6 | Implement `_repair(text)`: strip whitespace → remove markdown fences → single quotes → double quotes → remove trailing commas | 5 |
| 4.7 | Implement `validate(value, metadata)`: thử `json.loads()` → nếu lỗi thì gọi `_repair()` rồi thử lại → `PassResult` nếu thành công / `FailResult` nếu thất bại | 2 |
| 4.8 | Test với 5 test cases: Valid JSON, Markdown fences, Single quotes, Trailing comma, Truly invalid | 2 |

**Lưu ý quan trọng:**
- `on_fail=OnFailAction.FIX` truyền vào **validator constructor**: `PIIDetector(on_fail=OnFailAction.FIX)`
- KHÔNG truyền vào `Guard.use()`: ~~`Guard().use(PIIDetector, on_fail=...)`~~ ← SAI

**Xác minh:**
- PII bị phát hiện và redacted
- JSON lỗi được sửa thành công
- JSON không thể sửa trả về error object

---

## EVIDENCE — THU THẬP BẰNG CHỨNG

### Cách thu thập

| File | Cách tạo |
|------|----------|
| `01_langsmith_traces.png` | Chụp màn hình LangSmith UI → tab Run, hiển thị ≥ 50 traces |
| `02_prompt_hub.png` | Chụp màn hình LangSmith → tab Prompt Hub, hiển thị 2 versions |
| `02_ab_routing_log.txt` | `python 02_prompt_hub_ab_routing.py \| tee evidence/02_ab_routing_log.txt` |
| `03_ragas_scores.png` | Chụp terminal hiển thị bảng so sánh V1 vs V2 |
| `03_ragas_report.json` | `cp data/ragas_report.json evidence/03_ragas_report.json` |
| `04_pii_demo_log.txt` | `python 04_guardrails_validator.py \| tee evidence/04_pii_demo_log.txt` (chỉ phần PII) |
| `04_json_demo_log.txt` | `python 04_guardrails_validator.py \| tee evidence/04_json_demo_log.txt` (chỉ phần JSON) |

**Thực tế:** Chạy `04_guardrails_validator.@py` 1 lần, output chứa cả PII và JSON demo. Có thể tách thủ công hoặc lưu 1 file chung rồi đổi tên.

### evidence/README.md (optional, +1 bonus)
- Phân tích ngắn gọn: V1 hay V2 tốt hơn? Tại sao?
- Có thể giải thích: V2 (structured) thường cho faithfulness cao hơn vì context được xử lý kỹ hơn

---

## BONUS — TỐI ĐA +10 ĐIỂM

### Bonus từ rubric

| Bonus | Cách đạt |
|-------|----------|
| Faithfulness ≥ 0.9 cho cả 2 versions (+3) | Tinh chỉnh chunk_size, chunk_overlap, system prompt |
| Phân tích V1 vs V2 (+2) | Viết vào `evidence/README.md` |
| Code sạch, có docstring (+2) | Thêm docstring ngắn gọn cho mỗi hàm |
| Tất cả steps chạy qua `run_all.py` (+2) | Implement `run_all.py` đầy đủ |
| Error handling (+1) | Try/except cho API calls, fallback cho Prompt Hub |

---

## THỨ TỰ TRIỂN KHAI

```
1. config.py              ← Nền tảng, validate môi trường
2. qa_pairs.py            ← Dữ liệu dùng chung cho step 1-3
3. data/knowledge_base.txt ← Dữ liệu nguồn cho RAG
4. .env + .gitignore      ← Bảo mật
5. 01_langsmith_rag_pipeline.py ← Step 1 (nền tảng cho step 2-3)
6. 02_prompt_hub_ab_routing.py  ← Step 2 (phụ thuộc step 1)
7. 03_ragas_evaluation.py       ← Step 3 (phụ thuộc step 1-2, chạy lâu nhất)
8. 04_guardrails_validator.py   ← Step 4 (độc lập nhất)
9. run_all.py                   ← Orchestrator
10. evidence/                    ← Thu thập sau khi tất cả steps chạy thành công
```

---

## CÁC LỖI THƯỜNG GẶP & CÁCH TRÁNH

| Lỗi | Hậu quả | Cách tránh |
|-----|---------|------------|
| Commit `.env` | −10 điểm | Thêm vào `.gitignore` trước khi `git add` |
| `LANGCHAIN_TRACING_V2` set sau import | Không có traces | Set `os.environ` ở đầu file, trước tất cả imports |
| `on_fail` truyền vào `Guard.use()` | TypeError | Luôn truyền vào validator constructor |
| `retrieved_contexts` là string thay vì list | RAGAS context_recall = 0 | Giữ nguyên list of strings, không join |
| Import metrics từ `ragas.metrics.collections` | ImportError | Dùng `from ragas.metrics import faithfulness, ...` |
| Không dùng `numpy.mean()` | Gán list cho float | `float(np.mean(result['faithfulness']))` |
| Routing random (dùng `random.choice`) | −5 điểm | Dùng `hashlib.md5`, đảm bảo deterministic |
| Prompt không thực sự khác nhau | −5 điểm | V1: ngắn gọn 2-4 câu, V2: chi tiết 3-5 câu có cấu trúc |

---

## KIỂM TRA CUỐI CÙNG

Trước khi submit, chạy checklist:

- [ ] `python config.py` → ✅ Config loaded successfully
- [ ] `python 01_langsmith_rag_pipeline.py` → không lỗi, in 50 câu trả lời
- [ ] LangSmith UI có ≥ 50 traces
- [ ] `python 02_prompt_hub_ab_routing.py` → không lỗi, log mix V1/V2
- [ ] LangSmith Prompt Hub có 2 versions
- [ ] LangSmith UI có ≥ 100 traces tổng
- [ ] `python 03_ragas_evaluation.py` → không lỗi, faithfulness ≥ 0.8
- [ ] `data/ragas_report.json` tồn tại
- [ ] `python 04_guardrails_validator.py` → không lỗi, PII redacted, JSON repaired
- [ ] `evidence/` chứa đủ 7-8 file
- [ ] `.env` không có trong git history
- [ ] `git push origin main` thành công
