# CineSuggest

Trang gợi ý phim tiếng Việt dùng FastAPI, SQLite và embedding từ mô hình Hybrid GNN + SentenceTransformer đã huấn luyện trên MovieLens Latest-Small.
Danh mục, user MovieLens, số lượt đánh giá và rating trung bình được nhập từ dữ liệu small đã ghép nối. Tài khoản đăng ký trên website là hồ sơ riêng, không tạo ratings giả.

## Yêu cầu

- Python 3.10 trở lên.
- Dữ liệu catalog đã có trong `data/processed/`; model artifacts TMDB được đặt trong `model/web_artifacts_tmdb/` (có thể đổi bằng `MOVIE_MODEL_DIR`).
- Ollama và model `qwen2.5:1.5b` để dùng chatbot. Nếu PowerShell báo `ollama` không được nhận diện, hãy cài Ollama cho Windows từ [ollama.com/download/windows](https://ollama.com/download/windows), hoàn tất cài đặt rồi đóng/mở lại PowerShell hoặc VS Code để cập nhật `PATH`. Kiểm tra bằng `ollama --version`.

## Chạy local

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Mở Ollama từ Start menu để chạy dịch vụ nền. Nếu dịch vụ chưa chạy, mở một terminal riêng và chạy:

```powershell
ollama serve
```

Trong terminal khác, tải model:

```powershell
ollama pull qwen2.5:1.5b
```

Sau đó khởi chạy web app:

```powershell
uvicorn app:app --reload
```

Mở http://127.0.0.1:8000. Câu hỏi chatbot chỉ được gửi tới Ollama local; app không gọi dịch vụ LLM bên ngoài.

## Tài khoản admin demo

Mặc định không có admin dựng sẵn và đăng ký công khai chỉ tạo role `user`. Để tạo admin demo, đặt các biến môi trường trước khi khởi động:

```powershell
$env:ENABLE_ADMIN_DEMO = "true"
$env:ADMIN_EMAIL = "admin@example.test"
$env:ADMIN_PASSWORD = "set-a-unique-password-123"
uvicorn app:app --reload
```

Mật khẩu phải có ít nhất 10 ký tự, gồm chữ cái và chữ số. Dùng mật khẩu riêng và tắt `ENABLE_ADMIN_DEMO` sau khi tạo tài khoản. Không dùng tài khoản demo này trên môi trường internet.

Đặt `COOKIE_SECURE=true` khi phục vụ qua HTTPS. `MOVIE_DB_PATH` và `MOVIE_DATA_DIR` có thể đổi để chọn đường dẫn dữ liệu/SQLite khác.

## Tính năng và giới hạn

- Guest: xem phim điểm rating MovieLens cao và danh sách theo thể loại.
- User: đăng ký/đăng nhập và lưu thể loại/phim yêu thích. Khi có sở thích, mục “Dành cho bạn” tạo vector hồ sơ từ embedding phim yêu thích và thể loại, rồi xếp hạng các phim trong catalog có TMDB ID khớp với artifacts Hybrid GNN. User web mới không dùng embedding user MovieLens có sẵn; hồ sơ được dựng từ lựa chọn của họ. Nếu artifacts chưa sẵn sàng hoặc không có vector phù hợp, hệ thống lọc theo thể loại hoặc hiển thị phim rating cao.
- Admin: chỉnh role/trạng thái tài khoản, xóa user khác, sửa hoặc xóa phim trong catalog.
- Danh sách phim phân trang tối đa 10 phim mỗi trang. Mỗi thẻ hiển thị tên, thể loại, mô tả ngắn, rating trung bình và số lượt đánh giá; chọn thẻ để xem chi tiết.
- Chatbot: từ chối câu hỏi nhận diện được là ngoài phạm vi phim; LLM chỉ được chọn ID trong tập ứng viên truy xuất từ catalog. Mỗi phim được trình bày một dòng gồm tên, thể loại và lý do đề xuất ngắn; tên phim luôn lấy từ catalog, không dùng nội dung tên tự do do LLM sinh.
- Model artifacts TMDB được xuất từ notebook Hybrid GNN vào `model/web_artifacts_tmdb/`; web dùng trực tiếp `movie_emb.npy` đã lan truyền qua GNN để xếp hạng nhanh. `hybrid_gnn_llm_tmdb.pt` giữ state dict để tái tạo mô hình; web hiện không chạy lại SentenceTransformer/GraphSAGE cho mỗi request. `user_emb.npy` là embedding user trong tập train, không phải tài khoản đăng ký trên web.
- Website catalog và model đều dùng The Movies Dataset; phim được ghép qua TMDB ID. Chỉ phim có TMDB ID khớp mới được xếp hạng bằng embedding.
- `train_lightgcn.py` là pipeline LightGCN độc lập/legacy; web runtime dùng artifacts Hybrid GNN trong `model/`.
- `train_lightgcn.py` tạo `data/processed/movie_catalog.csv` để lưu metadata từng phim, gồm synopsis cho bước truy xuất chatbot.
- Khi database trống và dataset small có sẵn, startup nhập phim và ratings đã ghép vào SQLite. Nếu không có dữ liệu small thì danh sách/gợi ý trả về rỗng, không sinh rating giả.
- Admin quản lý tài khoản website; số user MovieLens nhập từ dataset được hiển thị riêng và không phải tài khoản đăng nhập.

Chạy kiểm thử:

```powershell
python -m unittest discover -s tests -v
```
