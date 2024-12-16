# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPool1D, Dropout, LayerNormalization
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Concatenate, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer
import pickle
import numpy as np

# Define the model architecture
def generate_model(data_vocab_size):
    dropout_threshold = 0.3  # Reduced dropout
    input_dim = data_vocab_size
    output_dim = 64  # Increased embedding dimension
    input_length = 512
    initializer = tf.keras.initializers.GlorotNormal()

    input_layer = Input(shape=(input_length,))
    feature = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length,
                        embeddings_initializer=initializer)(input_layer)

    # Convolutional Path
    cnn_feature = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(feature)  # Increased filters
    cnn_feature = MaxPool1D()(cnn_feature)  # Reduces to 256
    cnn_feature = Dropout(dropout_threshold)(cnn_feature)
    cnn_feature = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn_feature)  # Increased filters
    cnn_feature = MaxPool1D()(cnn_feature)  # Reduces to 128
    cnn_feature = LayerNormalization()(cnn_feature)
    cnn_feature = Dropout(dropout_threshold)(cnn_feature)

    # Recurrent Path
    bi_lstm_feature = Bidirectional(LSTM(units=64, dropout=dropout_threshold, return_sequences=True,
                                         kernel_initializer=initializer))(cnn_feature)  # Increased units
    bi_lstm_feature = MaxPool1D()(bi_lstm_feature)  # Reduces to 64
    bi_lstm_feature = Bidirectional(GRU(units=64, dropout=dropout_threshold, return_sequences=True,
                                        kernel_initializer=initializer))(bi_lstm_feature)  # Increased units
    bi_lstm_feature = MaxPool1D()(bi_lstm_feature)  # Reduces to 32
    bi_lstm_feature = LayerNormalization()(bi_lstm_feature)

    # Apply GlobalMaxPooling1D to both feature maps
    cnn_pooled = GlobalMaxPooling1D()(cnn_feature)          # (None, 64)
    bi_lstm_pooled = GlobalMaxPooling1D()(bi_lstm_feature)  # (None, 128)

    # Concatenate the pooled features
    combine_feature = Concatenate()([cnn_pooled, bi_lstm_pooled])  # (None, 192)
    combine_feature = LayerNormalization()(combine_feature)

    # Classification Layers
    classifier = Dense(128, activation='relu')(combine_feature)  # Increased units
    classifier = Dropout(0.2)(classifier)
    classifier = Dense(64, activation='relu')(classifier)  # Reduced units
    classifier = Dropout(0.2)(classifier)
    classifier = Dense(3, activation='softmax')(classifier)

    model = tf.keras.Model(inputs=input_layer, outputs=classifier)
    return model

# Function to preprocess raw input
def preprocess_raw_input(raw_input, tokenizer):
    # Tokenize the input text using ViTokenizer
    input_text_pre = ViTokenizer.tokenize(raw_input)
    st.write("Text preprocessed:", input_text_pre)
    # Convert text to sequences
    tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre])
    # Pad sequences
    vec_data = pad_sequences(tokenized_data_text, padding='post', maxlen=512)
    return vec_data

# Function to perform inference
def inference_model(input_feature, model):
    output = model.predict(input_feature)[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực'}
    label = label_dict.get(result, "Unknown")
    return label, conf

# Function to make prediction
def prediction(raw_input, tokenizer, model):
    input_model = preprocess_raw_input(raw_input, tokenizer)
    result, conf = inference_model(input_model, model)
    return result, conf

# Streamlit App
def main():
    st.title("Phân tích cảm xúc bằng mô hình CNN-BiLSTM")
    st.write("Nhập văn bản để phân loại cảm xúc thành Tiêu cực, Trung lập hoặc Tích cực.")

    # Text input
    raw_input = st.text_area("Văn bản đầu vào:", "Cô dạy rất tốt")

    # Load tokenizer
    try:
        with open("tokenizer_data.pkl", "rb") as input_file:
            tokenizer = pickle.load(input_file)
    except FileNotFoundError:
        st.error("Không tìm thấy tệp tokenizer_data.pkl. Vui lòng đảm bảo tệp này tồn tại.")
        return

    # Load model
    data_vocab_size = 3814  # Replace with your actual vocab size if different
    model = generate_model(data_vocab_size)
    try:
        model.load_weights("model_cnn_bilstm.keras")
    except FileNotFoundError:
        st.error("Không tìm thấy tệp model_cnn_bilstm.keras. Vui lòng đảm bảo tệp này tồn tại.")
        return

    # Compile the model (necessary before prediction if not compiled)
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Prediction button
    if st.button("Phân loại"):
        if raw_input.strip() == "":
            st.warning("Vui lòng nhập văn bản để phân loại.")
        else:
            with st.spinner("Đang xử lý..."):
                result, confidence = prediction(raw_input, tokenizer, model)
                st.success(f"Kết quả: **{result}** với độ tin cậy **{confidence*100:.2f}%**")

    # Example prediction on load (optional)
    if st.checkbox("Hiển thị ví dụ"):
        example_text = "Cô dạy rất tốt"
        st.write(f"**Ví dụ:** {example_text}")
        example_result, example_confidence = prediction(example_text, tokenizer, model)
        st.write(f"Kết quả: **{example_result}** với độ tin cậy **{example_confidence*100:.2f}%**")

if __name__ == "__main__":
    main()
