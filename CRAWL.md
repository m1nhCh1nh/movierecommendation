# Báo cáo thu thập dữ liệu

## Trạng thái

- Nguồn: [The Movies Dataset trên Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- Chủ sở hữu: Rounak Banik
- Phiên bản đã tải: **7**
- Ngày tải: **2026-11-21**
- Giấy phép hiển thị trên Kaggle: **CC0: Public Domain**
- Archive: `data/the-movies-dataset.zip` (238,862,293 bytes)
- Dữ liệu đã giải nén: `data/the-movies-dataset/` (943,755,800 bytes)
- SHA-256 của archive: `2257306173ac5ccd56a30737dee19452c6dd23b0a4cd21e30d4c266dcd2b04e6`
- Kiểm tra ZIP: **thành công** (`testzip` không phát hiện lỗi)

Archive và các CSV gốc được giữ nguyên. Bước crawl không thay đổi dữ liệu nguồn; phần ghép nối và thử nghiệm LightGCN được ghi riêng ở mục dưới.

## Các tệp đã thu thập

Số dòng dưới đây không tính dòng tiêu đề; dung lượng là byte của CSV đã giải nén.

| Tệp | Dòng dữ liệu | Dung lượng | Mục đích |
|---|---:|---:|---|
| `ratings.csv` | 26,024,289 | 709,550,327 | Ratings đầy đủ; dữ liệu tương tác chính |
| `ratings_small.csv` | 100,004 | 2,438,266 | Tập ratings nhỏ để bắt đầu kiểm thử pipeline sau này |
| `movies_metadata.csv` | 45,466 | 34,445,126 | Metadata phim, bao gồm `id`, `title`, `genres`, `overview` |
| `links.csv` | 45,843 | 989,107 | Ánh xạ MovieLens `movieId` sang TMDB `tmdbId` và IMDb |
| `links_small.csv` | 9,125 | 183,372 | Bảng liên kết tương ứng cho tập nhỏ |
| `keywords.csv` | 46,419 | 6,231,943 | Từ khóa phim dạng chuỗi JSON |
| `credits.csv` | 45,476 | 189,917,659 | Cast và crew dạng chuỗi JSON |

## Kiểm tra sơ bộ dữ liệu gốc

Các số liệu sau được thống kê trực tiếp từ CSV đã giải nén; không ghi thay đổi vào dữ liệu:

- `ratings.csv`: 270,896 user và 45,115 `movieId` khác nhau.
- `ratings_small.csv`: 671 user và 9,066 `movieId` khác nhau. Vì vậy, mô tả “700 users, 9.000 phim” chỉ là số làm tròn.
- `movies_metadata.csv`: có 3 dòng `id` không chuyển được sang số và 30 dòng trùng ID số. Pipeline loại các ID lỗi và giữ dòng đầu tiên cho mỗi ID trùng.
- `links.csv` có 219 dòng thiếu `tmdbId`; `links_small.csv` có 13 dòng thiếu `tmdbId`.
- Khóa phim giữa ratings và metadata không tương đương trực tiếp: cần nối `ratings.movieId` → `links.movieId` → `links.tmdbId` → `movies_metadata.id`. Dùng `links.csv` với ratings đầy đủ và `links_small.csv` với `ratings_small.csv`.
- `genres` là dữ liệu dạng chuỗi JSON/list; pipeline parse an toàn thành `genre_list`.
- Dữ liệu đầy đủ lớn hơn nhiều so với tập nhỏ: riêng `ratings.csv` đã khoảng 710 MB; tổng CSV sau giải nén khoảng 944 MB. Nên dùng tập nhỏ để kiểm thử bước tiền xử lý trước và cân nhắc giới hạn bộ nhớ khi xây dựng pipeline cho tập đầy đủ.

## Ghép nối và huấn luyện LightGCN

- Script: `train_lightgcn.py`
- Cài dependencies: `python -m pip install -r requirements.txt`
- Lệnh chạy: `python train_lightgcn.py --epochs 30`
- Ghép `ratings_small.csv` → `links_small.csv` → `movies_metadata.csv`; ratings chưa ghép được vẫn được giữ trong CSV đầu ra với `metadata_matched=false`.
- Dùng `rating >= 3.5` làm tương tác dương, re-index user/item liên tục và hold out một tương tác dương cho mỗi user làm validation (giữ mỗi item có ít nhất một cạnh trong tập train).
- Huấn luyện LightGCN 3 lớp, embedding 64 chiều, BPR loss, 30 epochs, seed 42 trên CPU; checkpoint lưu epoch tốt nhất theo NDCG@10 validation.
- Chỉ ratings dương tạo cạnh đồ thị; title/genres/overview là metadata để tra cứu và hiển thị, không được đưa vào LightGCN.
- Kết quả: 99,810/100,004 ratings ghép được metadata; 61,958 tương tác dương, chia thành 61,287 train và 671 validation; 671 users, 6,961 items.
- Validation tốt nhất: Recall@10 **0.1073**, NDCG@10 **0.0581** tại epoch 11. Đây là kết quả holdout đơn giản trên tập nhỏ, chỉ để xác nhận pipeline chạy được, chưa phải đánh giá production.

Các đầu ra trong `data/processed/`:

- `ratings_small_joined.csv`: ratings kèm TMDB ID, title, genres và trạng thái ghép.
- `movie_catalog.csv`: một dòng/phim đã ghép metadata, gồm overview để guest/admin/chatbot tra cứu.
- `lightgcn_checkpoint.pt`: trọng số epoch tốt nhất cùng cấu hình và ánh xạ user/item.
- `user_index.csv`, `item_index.csv`: ánh xạ ID gốc sang chỉ số mô hình; item index giữ thêm overview cho bước LLM sau này.
- `recommendations_top10.csv`: tối đa 10 phim chưa được user đánh giá trong dữ liệu đã ghép, cho mỗi user.
- `training_metrics.json`: thống kê ghép nối, cấu hình, lịch sử loss và metric validation.

Chưa huấn luyện LLM, lọc 10-core, hoặc dùng `ratings.csv` đầy đủ.

## Tham chiếu nguồn

Thông tin phiên bản, kích thước khai báo và giấy phép được lấy từ metadata Kaggle tại thời điểm thu thập. Bộ dữ liệu tổng hợp metadata TMDB cùng ratings/links MovieLens; cần giữ attribution và điều khoản của các nguồn upstream khi sử dụng hoặc phân phối dữ liệu.
