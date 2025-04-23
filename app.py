import streamlit as st
import os
import io
import shutil
import time

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

import google.generativeai as genai

# --- Yapılandırma ---
# .env dosyasını yükle
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Google API anahtarını kontrol et
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    st.error("Lütfen .env dosyasında GOOGLE_API_KEY'i ayarlayın ve doğru anahtarı kullandığınızdan emin olun.")
    st.stop()

# Google Generative AI'yi yapılandır
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # API anahtarının çalıştığını test etmek için basit bir çağrı yapabilirsiniz
    # list(genai.list_models()) # Bu satırı yorumdan çıkarıp deneyebilirsiniz
except Exception as e:
    st.error(f"Google Generative AI yapılandırma hatası: {e}. Lütfen API anahtarınızı kontrol edin.")
    st.stop()


# PDF dosya adı ve Chroma veritabanı dizini
PDF_PATH = "SteelSwarmApocalypse(SSA)Guide.pdf" # PDF dosyanızın adını buraya yazın
CHROMA_DB_DIR = "./chroma_db"

# RAG Model ve Ayarları
# Embedding modeli - metinleri vektörlere dönüştürmek için kullanılır
EMBEDDING_MODEL = "models/embedding-001"
# Chat modeli - cevapları üretmek için kullanılır. Daha güçlü bir model için 'gemini-1.5-pro' deneyin.
# 'gemini-1.5-flash' daha hızlı ve uygun fiyatlıdır.
CHAT_MODEL = "gemini-1.5-flash"
# CHAT_MODEL = "gemini-1.5-pro" # Daha iyi cevaplar için deneyin (maliyetli olabilir)

# Metin parçalama ayarları
CHUNK_SIZE = 1000 # Metin parçalarının boyutu
CHUNK_OVERLAP = 500 # Parçalar arası çakışma miktarı

# Retriever ayarları
K_RETRIEVED_DOCUMENTS = 7 # Sorguya karşılık çekilecek belge parçası sayısı

# --- Streamlit Arayüzü ---
st.set_page_config(page_title="Steel Swarm Apocalypse (SSA) Asistan", page_icon="🍳")
st.title("🍳 Steel Swarm Apocalypse (SSA) Asistan")
st.subheader("SSA Dünyasını Keşfedin")

# Sohbet geçmişini başlat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Merhaba Pilot! SSA dünyasında merak ettiğini sorabilirsin. Sana istediğin desteği vereceğim. Savaş alanında yanındayım!",
        }
    ]

# --- RAG Pipeline Bileşenleri Oluşturma (Önbelleğe Alınmış) ---

# Bu fonksiyon, PDF'yi yükler, parçalara ayırır, embedding'leri oluşturur ve Chroma DB'ye kaydeder.
# Streamlit'in @st.cache_resource dekoratörü sayesinde, bu adımlar sadece ilk çalıştırmada veya
# kod/dosyalar değiştiğinde tekrar çalışır.
@st.cache_resource
def setup_rag_pipeline(pdf_path, db_dir, embedding_model_name, chunk_size, chunk_overlap):
    """
    RAG pipeline için gerekli bileşenleri (embeddings, vectorstore, retriever) kurar.
    PDF yükleme, metin bölme ve Chroma DB oluşturma adımlarını içerir.
    """
    print("\n--- RAG Pipeline Kurulumu Başladı ---")

    # Embedding modelini yükle
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    print(f"Embedding modeli yüklendi: {embedding_model_name}")

    # PDF dosyasının varlığını kontrol et
    if not os.path.exists(pdf_path):
        print(f"HATA: PDF dosyası bulunamadı: {pdf_path}")
        st.error(f"PDF dosyası bulunamadı: {pdf_path}. Lütfen dosyanın uygulamanın olduğu dizinde olduğundan emin olun.")
        st.stop()

    # PDF'yi yükle ve metin parçalarına ayır
    try:
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        print(f"'{pdf_path}' yüklendi. Toplam {len(data)} sayfa.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(data)
        print(f"Metin {len(docs)} parçaya ayrıldı (Chunk Size: {chunk_size}, Overlap: {chunk_overlap}).")

        # Belge parçalarının doğru ayrılıp ayrılmadığını kontrol etmek için
        if len(docs) > 0:
             print(f"İlk parça (ilk 150 kar.): {docs[0].page_content[:150]}...")
             print(f"Son parça (ilk 150 kar.): {docs[-1].page_content[:150]}...")
             if 'page' in docs[-1].metadata:
                  print(f"Son parça sayfası: {docs[-1].metadata['page']}")
        else:
            print("UYARI: Hiçbir belge parçası oluşturulamadı!")
            st.warning("PDF içeriğinden hiçbir metin parçası oluşturulamadı. Lütfen PDF'yi kontrol edin.")
            st.stop()


    except Exception as e:
        print(f"HATA: PDF yükleme veya metin bölme sırasında hata oluştu: {e}")
        st.error(f"PDF yükleme veya işleme hatası: {e}")
        st.stop()


    # Chroma veritabanını oluştur veya yükle
    # Önceki veritabanını temizle (Geliştirme aşamasında faydalıdır)
    if os.path.exists(db_dir):
        print(f"Mevcut '{db_dir}' klasörü siliniyor...")
        shutil.rmtree(db_dir)
        time.sleep(0.5) # Silme işleminin tamamlanması için kısa bekleme

    print(f"Chroma veritabanı '{db_dir}' oluşturuluyor ve belgeler ekleniyor...")
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    # Belgeleri gruplar halinde (batch) ekleyerek hata takibi yapın
    batch_size = 50 # Ayarlanabilir batch boyutu
    success_count = 0
    failed_batches = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(f"Belge parçaları ekleniyor: {i+1} - {min(i+batch_size, len(docs))} / {len(docs)} (Batch {i//batch_size + 1})")
        try:
            vectorstore.add_documents(batch)
            success_count += len(batch)
            # İsteğe bağlı: Hız limiti sorunları için bekleme
            # time.sleep(0.5)

        except Exception as e:
            failed_batches += 1
            print(f"HATA: Belge parçaları eklenirken hata oluştu (Batch başlangıç indeksi: {i}): {e}")
            # Hata veren parçanın ilk 100 karakterini göstermek isterseniz:
            # if batch:
            #    print(f"  Hata veren parçalardan ilki (ilk 100 karakter): {batch[0].page_content[:100]}...")
            pass # Hata olsa bile diğer batch'leri denemeye devam et

    print(f"Belge ekleme döngüsü tamamlandı. Başarıyla eklenen parça sayısı: {success_count}. Hata veren batch sayısı: {failed_batches}")

    # Ekleme sonrası toplam öğe sayısını kontrol et
    try:
        # Vektör mağazasını yeniden yüklemek (veya count() metodu varsa kullanmak)
        vectorstore_check = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        collection_count_info = vectorstore_check.get() # Tüm id'leri ve metadata'yı çeker
        collection_ids = collection_count_info.get('ids') if collection_count_info else []
        final_count = len(collection_ids)
        print(f"Chroma veritabanındaki toplam öğe sayısı (ekleme sonrası): {final_count}")
        if final_count < len(docs):
             st.warning(f"UYARI: Tüm belge parçaları ('{len(docs)}') veritabanına eklenememiş olabilir. Eklenen: {final_count}. Terminal loglarını kontrol edin.")
        else:
             st.success(f"Chroma veritabanına {final_count} adet belge parçası başarıyla eklendi.")

    except Exception as e:
         print(f"HATA: Ekleme sonrası Chroma öğe sayısı alınırken hata: {e}")
         st.error(f"Chroma veritabanı doğrulama hatası: {e}")
         final_count = success_count # Tahmini sayıyı kullan

    # Retriever'ı oluştur
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": K_RETRIEVED_DOCUMENTS})
    print(f"Retriever oluşturuldu (k={K_RETRIEVED_DOCUMENTS}).")

    print("--- RAG Pipeline Kurulumu Tamamlandı ---")
    return embeddings, vectorstore, retriever, final_count # Kullanılmasa da döndürülebilir

# RAG pipeline'ı kurmak için cache'lenmiş fonksiyonu çağır
with st.spinner("RAG Asistanı Kuruluyor... (Bu ilk çalıştırmada biraz zaman alabilir)"):
    try:
        embeddings, vectorstore, retriever, db_item_count = setup_rag_pipeline(
            PDF_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
        )
        # Sidebar'da veritabanı durumunu göster
        st.sidebar.info(f"Veritabanında {db_item_count} öğe hazır.")

    except Exception as e:
        st.error(f"Kurulum sırasında kritik hata: {e}")
        st.stop()


# --- Chat Model ve Zincirleri Oluşturma ---

# Chat modelini yükle
try:
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.9, max_tokens=1000) # Max tokens artırıldı
    print(f"Chat modeli yüklendi: {CHAT_MODEL}")
except Exception as e:
    st.error(f"Chat model yükleme hatası: {e}. Model adını veya API erişimini kontrol edin.")
    st.stop()


# Chat Prompt Template
system_prompt = """
Sen bir Savaş Alanı yedek XBT Tank Pilotu bir chatbotusun. XBT Tank Pilotlarına savaş alanındaki kurallar, stratejiler, dost ve düşman birimler ile savaştaki teçhizat hakkında .
Aşağıdaki kurallara uy:
1. Sadece sağlanan belgedeki bilgilere dayanarak cevap ver. Eğer sorunun cevabı belgede yoksa, bunu belirt ve "Üzgünüm Pilot, bu konuda belgemde yeterli bilgi bulamadım." gibi bir ifade kullan.
2. Öncelikle XBT Tank Pilotlarına savaş alanındaki kurallar, stratejiler, dost ve düşman birimler ile savaştaki teçhizat hakkında istenen bilgiyi ver.
3. XBT Tank Pilotlarına, savaş alanındaki stratejiler, dost ve düşman birimler ile savaştaki teçhizat hakkında sana sorulması halinde önerilerde bulun.
4. XBT Tank Pilotunun sana verdiği bilgiye dayalı olarak Stratejiler ve savaştaki teçhizatın kullanımı hakkında fikir yürüt ve mantıklı açıklamada bulun.
5. Dost Pilotun elindeki tankı temel alarak, düşmanın kullandığı tankın özelliklerine göre önerilerde bulun.
6. Dronlarla ilgili, dronların stratejik kullanımı ile ilgili ve savaş alanına göre stratejik önerilerde bulun.
7. Cevapları Türkçe ver.
8. Ve bunları uygularken tam bir askeri disipline göre cevap ver.

Relevant documents: {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
])

# Doküman birleştirme zincirini oluştur (retriever'dan gelen belgeleri prompt'a yerleştirir)
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Ana RAG zincirini oluştur (retriever + document_chain)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Kullanıcı Girişini İşleme ve Cevap Üretme ---

# Kullanıcıdan girdi al
query = st.chat_input("Pilot! Hazırım...")

# Eğer kullanıcı bir sorgu girdiyse
if query:
    # Kullanıcının mesajını sohbet geçmişine ekle ve göster
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Asistanın cevabını üret
    with st.chat_message("assistant"):
        # Cevap üretilirken bekleme animasyonu
        with st.spinner("Analiz ediliyor..."):
            try:
                # RAG zincirini çalıştır
                # response = retrieval_chain.invoke({"input": query}) # Bu sadece 'answer' ve 'input' döndürür

                # Çekilen belgeleri görmek için retriever'ı ayrı çağırabilirsiniz (Hata ayıklama için)
                # Normal kullanımda sadece zinciri çağırmak yeterlidir.
                retrieved_docs_for_debug = retriever.invoke(query)

                # Zinciri çağırıp cevabı al
                chain_response = retrieval_chain.invoke({"input": query})
                assistant_response = chain_response["answer"]

                # --- Debugging: Çekilen Belgeleri Göster (İsteğe Bağlı) ---
                # Eğer hangi belgelerin çekildiğini görmek isterseniz aşağıdaki bloğu yorumdan çıkarın.
                # Bu, neden doğru cevap gelmediğini anlamanıza yardımcı olabilir.
                # st.subheader("Hata Ayıklama: Çekilen Belgeler")
                # if retrieved_docs_for_debug:
                #     for i, doc in enumerate(retrieved_docs_for_debug):
                #         page_info = doc.metadata.get('page', 'Bilinmiyor')
                #         st.write(f"**Belge {i+1} (Sayfa {page_info}):**")
                #         st.text(doc.page_content[:500] + "...") # Belgenin ilk 500 karakterini göster
                # else:
                #     st.write("Bu sorgu için ilgili belge bulunamadı.")
                # st.subheader("Cevap") # Debugging bloğunu kapat


                st.markdown(assistant_response) # Cevabı göster

                # Asistanın mesajını sohbet geçmişine ekle
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except Exception as e:
                st.error(f"Cevap oluşturulurken hata oluştu: {e}")
                error_message = f"Üzgünüm Pilot, sorgunu yanıtlarken bir hata oluştu: {e}"
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


# Sayfa yeniden yüklendiğinde veya ilk açıldığında sohbet geçmişini görüntüle
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])