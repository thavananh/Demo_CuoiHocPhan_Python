# Phân Tích Cảm Xúc Bằng Mô Hình CNN-BiLSTM

Ứng dụng **Phân Tích Cảm Xúc Bằng Mô Hình CNN-BiLSTM** là một ứng dụng web được xây dựng bằng Streamlit, sử dụng mô hình học sâu kết hợp giữa Convolutional Neural Network (CNN) và Bidirectional LSTM (BiLSTM) để phân loại cảm xúc của văn bản tiếng Việt thành ba loại: **Tiêu cực**, **Trung lập**, và **Tích cực**.

## Tính Năng

- **Phân loại cảm xúc**: Nhập văn bản tiếng Việt và nhận diện cảm xúc tương ứng.
- **Giao diện thân thiện**: Sử dụng Streamlit để tạo giao diện đơn giản và dễ sử dụng.
- **Hiển thị độ tin cậy**: Kết quả phân loại đi kèm với mức độ tin cậy của mô hình.
- **Ví dụ mẫu**: Hiển thị ví dụ phân loại khi người dùng muốn.

## Yêu Cầu Hệ Thống

- **Python**: Phiên bản 3.7 trở lên
- **Các thư viện Python**: Xem mục [Cài Đặt](#cài-đặt) để biết chi tiết.

## Cài Đặt

1. **Clone kho lưu trữ**:

    ```bash
    git clone https://github.com/thavananh/Demo_CuoiHocPhan_Python.git
    cd Demo_CuoiHocPhan_Python
    ```

2. **Tạo môi trường ảo (khuyến nghị)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3. **Cài đặt các thư viện phụ thuộc qua `requirements.txt`**:

    Đảm bảo bạn đã có tệp `requirements.txt` trong thư mục dự án. Tệp này liệt kê tất cả các thư viện cần thiết để chạy ứng dụng. Nội dung mẫu của `requirements.txt`:

    ```plaintext
    streamlit
    tensorflow
    pyvi
    numpy
    pickle5
    ```

    Sau đó, chạy lệnh sau để cài đặt tất cả các thư viện:

    ```bash
    pip install -r requirements.txt
    ```

4. **Tải mô hình và tokenizer**:

    Đảm bảo bạn có các tệp sau trong thư mục dự án:
    
    - `model_cnn_bilstm.keras`: Tệp trọng số của mô hình.
    - `tokenizer_data.pkl`: Tệp tokenizer đã được huấn luyện.

    Nếu bạn chưa có các tệp này, hãy liên hệ với tác giả hoặc kiểm tra tài liệu đi kèm để biết cách tạo và lưu trữ chúng.

## Sử Dụng

### Chạy Ứng Dụng

Để chạy ứng dụng Streamlit, sử dụng lệnh sau:

```bash
streamlit run app.py
