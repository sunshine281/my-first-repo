# ============================================================
# 预测性维护知识问答助手 —— 基于 RAG（检索增强生成）架构
# 核心流程：用户提问 → 向量检索相关文档 → Prompt 增强 → LLM 生成答案
# ============================================================

# ---------- 轻量 import（尽量让首屏快速出来） ----------
import os, sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------- 全局配置 ----------
_BASE_DIR = Path(__file__).resolve().parent

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

_env_docs = os.getenv("DOCS_DIR", "docs")
_env_chroma = os.getenv("CHROMA_DB_DIR", "chroma_db")

DOCS_DIR = _env_docs if Path(_env_docs).is_absolute() else str(_BASE_DIR / _env_docs)
CHROMA_DB_DIR = _env_chroma if Path(_env_chroma).is_absolute() else str(_BASE_DIR / _env_chroma)

# ---------- 启动校验 ----------
if not DEEPSEEK_API_KEY:
    st.set_page_config(page_title="预测性维护问答助手", layout="wide")
    st.error("❌ 请在 .env 文件中设置 DEEPSEEK_API_KEY")
    st.stop()

# ---------- Streamlit 页面设置（尽可能靠前，让浏览器立刻渲染首屏） ----------
st.set_page_config(page_title="预测性维护问答助手", layout="wide")
st.title("🔧 预测性维护知识问答助手")
st.markdown("基于您的内部文档，回答设备维护相关问题，并自动标注引用来源。")

# ---------- Embedding 模型缓存 ----------
@st.cache_resource
def get_embedding_model():
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"嵌入计算设备: {device.upper()}")
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------- 向量库初始化 ----------
@st.cache_resource
def init_vectorstore():
    from langchain_chroma import Chroma

    if os.path.exists(CHROMA_DB_DIR):
        st.info("从磁盘加载已有向量库")
        embedding_model = get_embedding_model()
        return Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model,
        )

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        st.error(f"文档目录 '{DOCS_DIR}' 不存在，请创建并放入 PDF/TXT 文件。")
        return None

    loaders = [
        DirectoryLoader(str(docs_path), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(docs_path), glob="**/*.txt", loader_cls=TextLoader),
    ]
    documents = []
    with st.spinner("正在加载文档..."):
        for loader in loaders:
            documents.extend(loader.load())

    if not documents:
        st.warning("未找到任何文档，请确认目录中有 PDF 或 TXT 文件。")
        return None
    st.info(f"已加载 {len(documents)} 个文档")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"共生成 {len(chunks)} 个切片")

    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR,
    )
    st.success("向量库已构建并持久化")
    return vectorstore

vectorstore = init_vectorstore()
if vectorstore is None:
    st.stop()

# ---------- 检索器 ----------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- LLM（懒初始化） ----------
@st.cache_resource
def get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="deepseek-flash",
        temperature=0.1,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
    )

# ---------- RAG 主函数 ----------
def ask_question(question: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    docs = retriever.invoke(question)
    context_parts = []
    references = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知文档")
        page = doc.metadata.get("page", "")
        context_parts.append(
            f"[{i+1}] 来源：《{Path(source).name}》第{page}页\n{doc.page_content}"
        )
        references.append(f"[{i+1}] 《{Path(source).name}》第{page}页")
    context = "\n\n".join(context_parts)

    system_prompt = """你是一位专业的预测性维护工程师。请严格依据以下提供的文档内容回答问题。
如果文档中没有足够的信息，请诚实地说"文档中未找到相关信息"。不要在答案中添加虚构的内容。"""
    user_prompt = f"""以下是相关文档片段：
{context}

用户问题：{question}

请给出回答："""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = get_llm().invoke(messages)

    references_md = "\n".join(references)
    return f"{response.content}\n\n---\n📎 **引用来源：**\n{references_md}"

# ---------- UI ----------
st.subheader("💬 开始提问")
with st.form(key="qa_form"):
    question = st.text_input(
        "请输入您的问题（例如：滚动轴承的振动加速度超过多少需要预警？）"
    )
    submit_button = st.form_submit_button(label="提交")

if submit_button and question.strip():
    with st.spinner("正在检索并生成答案..."):
        try:
            answer = ask_question(question)
            st.markdown("### 📝 答案")
            st.write(answer)
        except Exception as e:
            st.error(f"发生错误：{e}")
elif submit_button:
    st.warning("请输入问题后再提交。")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📂 文档列表")
    docs_path = Path(DOCS_DIR)
    if docs_path.exists():
        files = list(docs_path.glob("*.*"))
        if files:
            for f in files:
                st.write(f"- {f.name}")
        else:
            st.write("暂无文档")
    else:
        st.write("文档目录不存在")
        print('666')