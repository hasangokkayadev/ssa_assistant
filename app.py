import io
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    st.error("Lütfen .env dosyasında GOOGLE_API_KEY'i ayarlayın.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Steel Swarm Apocalypse (SSA) Asistan", page_icon="🍳")
st.title("🍳 Steel Swarm Apocalypse (SSA) Asistan")
st.subheader("SSA Dünyasını Keşfedin")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Merhaba Pilot! SSA dünyasında merak ettiğini sorabilirsin. Sana istediğin desteği vereceğim. Savaş alanında yanındayım",
        }
    ]


loader = PyPDFLoader("SteelSwarmApocalypse(SSA)Guide.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs= {"k" : 10})

#UYGULAMAMIZIN RAG SAFHASINA GEÇİŞ YAPIYORUZ.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

query=st.chat_input("Pilot! Hazırım...")
prompt = query

system_prompt = """
Sen bir Savaş Alanı yedek XBT Tank Pilotu bir chatbotusun. XBT Tank Pilotlarına savaş alanındaki kurallar, stratejiler, dost ve düşman birimler ile savaştaki teçhizat hakkında .
Aşağıdaki kurallara uy:
1. Öncelikle XBT Tank Pilotlarına savaş alanındaki kurallar, stratejiler, dost ve düşman birimler ile savaştaki teçhizat hakkında istenen bilgiyi ver.
2. XBT Tank Pilotlarına, savaş alanındaki stratejiler, dost ve düşman birimler ile savaştaki teçhizat hakkında sana sorulması halinde önerilerde bulun.
3. XBT Tank Pilotunun sana verdiği bilgiye dayalı olarak Stratejiler ve savaştaki teçhizatın kullanımı hakkında fikir yürüt ve mantıklı açıklamada bulun.
4. Dost Pilotun elindeki tankı temel alarak, düşmanın kullandığı tankın özelliklerine göre önerilerde bulun.
5. Dronlarla ilgili, dronların stratejik kullanımı ile ilgili ve savaş alanına göre stratejik önerilerde bulun.
6. Cevapları Türkçe ver.
7. Ve bunları uygularken tam bir askeri disipline göre cevap ver.
"""

# 1. Prompt Template Oluşturma
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\nRelevant documents: {context}"),
    ("user", "{input}"),
])

# 2. Doküman Birleştirme Zinciri Oluşturma
document_chain = create_stuff_documents_chain(llm, prompt_template)

# 3. Ana RAG Zincirini Oluşturma
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. Kullanıcı Girdisini İşleme ve Cevabı Görüntüleme
if prompt:
    # Kullanıcının mesajını sohbet geçmişine ekle
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval zincirini çalıştır
        response = retrieval_chain.invoke({"input": prompt})
        # Cevabın sadece 'answer' kısmını al
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        # Asistanın mesajını sohbet geçmişine ekle
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Sohbet geçmişini görüntüleme (eğer kullanıcı yeni bir mesaj girmediyse sayfa yenilendiğinde geçmişi gösterir)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
