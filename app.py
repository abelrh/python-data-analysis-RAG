import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# إعداد واجهة المستخدم
st.set_page_config(page_title="PDF Data Analysis Chatbot", page_icon="🐼", layout="centered")
st.title("🐼 محرك الـ RAG للبحث في كتب البيانات")
st.write("ارفع كتاب 'Python for Data Analysis' أو أي ملف PDF وابدأ الشات مع الموديل Llama 3.3!")

# خانة إدخال الـ Groq API Key بأمان من الشريط الجانبي
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if not groq_api_key:
    # لتسهيل التجربة في الأول، لو المفتاح مش في الشريط الجانبي، هيقرا المفتاح الثابت بتاعك
    groq_api_key = "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"

os.environ["GROQ_API_KEY"] = groq_api_key

# رفع ملف الـ PDF من خلال الواجهة مباشرة بدلاً من كود الكولاب
uploaded_file = st.file_uploader("قم برفع ملف الـ PDF هنا 👇", type=["pdf"])

# دالة معالجة الـ PDF وبناء الـ Vector Store
@st.cache_resource
def initialize_rag(file_bytes, file_name):
    temp_path = f"temp_{file_name}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        return vector_store.as_retriever(search_kwargs={"k": 4})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if uploaded_file is not None:
    retriever = initialize_rag(uploaded_file.read(), uploaded_file.name)
    st.success("✅ تم تحميل الكتاب وتقسيمه بنجاح! يمكنك بدء المحادثة.")

    # تعريف الموديل والبرومبت الصارم
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    
    system_prompt = """You are an advanced expert assistant. Your job is to answer the user's question BASED ONLY on the provided context below.
    The context is extracted from a specific PDF book.
    
    Strict Rules:
    1. Look deeply into the context to find the answer.
    2. If the answer cannot be found in the context, strictly say: "I cannot find the answer to this question in the provided book."
    3. Do not use any external knowledge or make up facts.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(system_prompt)

    # حفظ حالة الشات وتاريخ المحادثة (Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # عرض الرسائل السابقة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # استقبال الأسئلة من المستخدم
    if user_query := st.chat_input("اسألني عن أي كود أو مفهوم داخل الكتاب..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("جاري قراءة صفحات الكتاب واستخراج الكود المناسب..."):
                docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = prompt_template.format(context=context_text, question=user_query)
                
                response = llm.invoke(final_prompt)
                bot_response = response.content
                st.markdown(bot_response)
                
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
