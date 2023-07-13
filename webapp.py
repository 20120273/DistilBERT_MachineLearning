import streamlit as st
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering,DistilBertForSequenceClassification
import torch

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
model_ans = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
model_cls = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-distilled-squad')
def answer_question(context, question):
    # Tokenize the input
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model_ans(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]), return_dict=False)

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer_tokens_to_string



def sentiment_ana(sentence):
    d = {1:'Đây là nhận xét tiêu cực',0:'Đây là nhận xét tích cực'}
    test_sample = tokenizer([sentence], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model_cls(**test_sample)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
    


# Streamlit web app
def main():
    
    page = st.sidebar.radio("Thanh điều hướng",["Home","Question Answering","Sentiment Analysis" ])
    if page == "Home":
        st.title("DEMO chức năng Question Answering và Sentiment Analysis của mô hình DistilBERT ")
        st.subheader("🖥🖥Sản phẩm phục vụ cho môn Nhập môn Học máy - 20-21")
        st.write("""
            Đây là sản phẩm của nhóm 23, gồm các thành viên:
                 
                - Phạm Quang Tân - MSSV: 20120184
                - Phạm Duy Trường - MSSV: 20120230
                - Lê Thừa Phương Cát - MSSV: 20120256
                - Bùi Hồng Dương - MSSV: 20120273
                - Nguyễn Đăng Khương - MSSV: 20120516
                 """)
        st.header('Giới thiệu sản phẩm')
        st.write("""
                Nhóm sinh viên sử dụng mô hình DistilBERT trong bài báo **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**
                 cho 2 tác vụ:

                - 🔑 Question answering: tìm kiếm trả lời cho câu hỏi được đặt ra dựa trên mẫu thông tin cung cấp
                - ☢️ Sentiment Analysis: phân tích câu nhận xét mang ý nghĩa tiêu cực hay tích cực.
                 """)

    if page == "Question Answering" :
        st.title("DistilBERT Question Answering")
        st.write("""
                 Tại đây, bạn sẽ nhập vào một đoạn thông tin (context) sau đó sẽ đặt một câu hỏi liên quan đến các thông tin trong đoạn đó.
                 Mô hình sẽ tìm ra câu trả lời ở trong đoạn thông tin trên.
                 """)

        # User input
        context = st.text_area("Nhập đoạn thông tin:")
        question = st.text_input("Nhập câu hỏi liên quan:")

        if st.button("Câu trả lời"):
            if context and question:
                answer = answer_question(context, question)
                st.write("Answer:", answer)
            else:
                st.write("Yêu cầu nhập đủ cả hai phần.")
    if page =="Sentiment Analysis":
        st.title("DistilBERT Sentiment Analysis")
        st.write("""
                 Tại đây, bạn sẽ nhập vào một câu nhận xét và mô hình sẽ phân tích xem câu nhận xét trên mang ý nghĩa tiêu cực hay tích cực.
                 """)
        sentence = st.text_area('Nhập vào câu nhận xét')
        if st.button("Phân tích"):
            if sentence:
                sentiment_ana(sentence)
            else:
                st.write("Vui lòng nhập câu nhận xét.")
            
if __name__ == '__main__':
    main()
