"""
Streamlit 用户界面：Prompt 助手
"""
import streamlit as st
import json
from ollama_client import OllamaClient
from vector_store import VectorStore
from rag_generator import RAGGenerator
from config import TOP_K


# 页面配置
st.set_page_config(
    page_title="PromptRAG - AI 绘图提示词助手",
    page_icon="🎨",
    layout="wide"
)

# 初始化 session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_generator' not in st.session_state:
    st.session_state.rag_generator = None
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = None


def init_components():
    """初始化组件"""
    try:
        if st.session_state.ollama_client is None:
            st.session_state.ollama_client = OllamaClient()
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
            if st.session_state.vector_store.exists():
                st.session_state.vector_store.load_index()
            else:
                st.warning("⚠️ 向量库不存在，请先构建索引")
                return False
        
        if st.session_state.rag_generator is None:
            st.session_state.rag_generator = RAGGenerator(
                st.session_state.vector_store,
                st.session_state.ollama_client
            )
        
        return True
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        return False


def main():
    """主界面"""
    st.title("🎨 PromptRAG - AI 绘图提示词助手")
    st.markdown("---")
    
    # 侧边栏：系统状态
    with st.sidebar:
        st.header("⚙️ 系统状态")
        
        # 测试连接
        if st.button("🔌 测试 Ollama 连接"):
            client = OllamaClient()
            if client.test_connection():
                st.success("✓ 连接成功")
            else:
                st.error("✗ 连接失败")
        
        st.markdown("---")
        
        # 向量库状态（使用缓存的实例，避免重复加载）
        if st.session_state.vector_store is not None:
            store = st.session_state.vector_store
            if store.exists():
                st.success("✓ 向量库已就绪")
                if store.index is not None:
                    st.info(f"📊 索引大小: {store.index.ntotal} 条")
            else:
                st.warning("⚠️ 向量库未构建")
        else:
            # 快速检查，不加载模型
            import os
            from config import INDEX_PATH
            if os.path.exists(INDEX_PATH):
                st.info("📊 向量库文件存在，等待初始化...")
            else:
                st.warning("⚠️ 向量库未构建")
        
        st.markdown("---")
        st.markdown("### 📖 使用说明")
        st.markdown("""
        1. 在输入框中输入你的创作想法（支持中文和英文）
        2. 系统会自动检索相似的历史 Prompt
        3. 点击生成，获得优化后的中文 Prompt
        4. 复制到 ComfyUI 使用
        """)
    
    # 主界面
    if not init_components():
        st.stop()
    
    # 搜索与生成区域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "💭 输入你的创作想法（支持中文和英文）",
            placeholder="例如：赛博朋克风格的雨夜猫咪，霓虹灯，未来感...",
            height=100
        )
    
    with col2:
        top_k = st.number_input("检索数量", min_value=1, max_value=10, value=TOP_K, step=1)
        generate_btn = st.button("🚀 生成 Prompt", type="primary", use_container_width=True)
    
    if generate_btn and user_input:
        with st.spinner("正在检索和生成..."):
            try:
                # 生成
                result = st.session_state.rag_generator.generate(user_input, top_k=top_k)
                
                # 显示结果
                st.markdown("---")
                st.subheader("✨ 生成的中文 Prompt")
                
                # 可复制的 Prompt 框
                st.code(result["final_prompt"], language="text")
                
                # 复制按钮（Streamlit 原生支持）
                st.markdown("💡 点击上方代码框右上角的复制按钮即可复制")
                st.info("📝 提示：生成的是中文提示词，可直接用于支持中文的 ComfyUI 工作流")
                
                # 显示参考素材
                with st.expander("📚 参考素材（展开查看）", expanded=False):
                    for i, ref in enumerate(result["references"], 1):
                        st.markdown(f"### 参考 {i}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**主体**: {ref.get('subject', 'N/A')}")
                            st.markdown(f"**风格**: {ref.get('art_style', 'N/A')}")
                            st.markdown(f"**氛围**: {ref.get('mood', 'N/A')}")
                        
                        with col_b:
                            st.markdown(f"**视觉元素**: {', '.join(ref.get('visual_elements', []))}")
                            st.markdown(f"**技术参数**: {', '.join(ref.get('technical', []))}")
                        
                        with st.expander("原始 Prompt"):
                            st.text(ref.get('raw', 'N/A'))
                        
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"生成失败: {str(e)}")
                st.exception(e)
    
    elif generate_btn:
        st.warning("请输入你的创作想法")
    
    # 快速检索功能
    st.markdown("---")
    st.subheader("🔍 快速检索")
    
    col3, col4 = st.columns([3, 1])
    with col3:
        search_query = st.text_input("搜索关键词", placeholder="输入关键词查看相似的历史 Prompt")
    with col4:
        search_btn = st.button("🔎 搜索", use_container_width=True)
    
    if search_btn and search_query:
        try:
            results = st.session_state.vector_store.search(search_query, top_k=5)
            
            if results:
                st.markdown(f"找到 {len(results)} 个相似结果：")
                for i, (metadata, distance) in enumerate(results, 1):
                    with st.expander(f"结果 {i} (相似度: {1/(1+distance):.2%})"):
                        st.json(metadata)
            else:
                st.info("未找到相关结果")
        except Exception as e:
            st.error(f"搜索失败: {str(e)}")


if __name__ == "__main__":
    main()

