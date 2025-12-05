"""
Streamlit 用户界面：Prompt 助手
"""
import streamlit as st
import json
import time
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
            # 使用占位符显示加载状态
            with st.spinner("正在初始化向量库..."):
                st.session_state.vector_store = VectorStore()
                if st.session_state.vector_store.exists():
                    st.session_state.vector_store.load_index()
                    # 预热 encoder（进行一次 encode，避免首次检索时慢）
                    try:
                        st.session_state.vector_store.encoder.encode(["预热"])
                    except:
                        pass
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
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            search_btn = st.button("🔍 仅检索", use_container_width=True, help="只执行检索，不生成")
        with col_btn2:
            generate_btn = st.button("🚀 生成", type="primary", use_container_width=True, help="检索 + 生成完整流程")
        # 模型预热按钮（可选）
        if st.button("🔥 模型预热", use_container_width=True, help="调用一次短请求，让模型常驻以降低 TTFT"):
            with st.spinner("正在预热模型..."):
                try:
                    st.session_state.rag_generator.client.generate(
                        prompt="说一句话：模型预热完成。",
                        system="你是一个简短回答助手，只需一句话。",
                        temperature=0.1,
                    )
                    st.success("✓ 预热完成，可降低首 token 延迟")
                except Exception as e:
                    st.error(f"预热失败: {e}")
    
    # 仅检索模式
    if search_btn and user_input:
        import time
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔍 正在检索...")
            progress_bar.progress(10)
            
            start_time = time.time()
            retrieved = st.session_state.vector_store.search(user_input, top_k=top_k)
            search_time = time.time() - start_time
            retrieved_items = [item for item, _ in retrieved]
            
            progress_bar.progress(100)
            status_text.text(f"✓ 检索完成！耗时: {search_time:.3f} 秒")
            
            # 显示检索结果
            st.markdown("---")
            st.subheader(f"🔍 检索结果（找到 {len(retrieved_items)} 条）")
            st.info(f"⏱️ 检索耗时: **{search_time:.3f} 秒**")
            
            for i, ref in enumerate(retrieved_items, 1):
                with st.expander(f"结果 {i}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**主体**: {ref.get('subject', 'N/A')}")
                        st.markdown(f"**风格**: {ref.get('art_style', 'N/A')}")
                        st.markdown(f"**氛围**: {ref.get('mood', 'N/A')}")
                    with col_b:
                        elements = ref.get('visual_elements', [])
                        tech = ref.get('technical', [])
                        st.markdown(f"**视觉元素**: {', '.join(elements[:5]) if elements else 'N/A'}")
                        st.markdown(f"**技术参数**: {', '.join(tech[:5]) if tech else 'N/A'}")
                    st.text(f"原始: {ref.get('raw', 'N/A')}")
            
            # 保存检索结果到 session state，供生成使用
            st.session_state.last_search_results = retrieved_items
            st.session_state.last_user_input = user_input
            
        except Exception as e:
            st.error(f"检索失败: {str(e)}")
            st.exception(e)
        finally:
            progress_bar.empty()
            status_text.empty()
    
    # 完整生成流程
    elif generate_btn and user_input:
        import time
        # 分步显示进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. 检索阶段
            status_text.text("🔍 步骤 1/2: 正在检索相似提示词...")
            progress_bar.progress(10)
            
            search_start = time.time()
            # 执行检索（encoder 已在初始化时预热，这里应该很快）
            retrieved = st.session_state.vector_store.search(user_input, top_k=top_k)
            search_time = time.time() - search_start
            retrieved_items = [item for item, _ in retrieved]
            
            progress_bar.progress(30)
            status_text.text(f"✓ 检索完成（耗时: {search_time:.3f}秒），找到 {len(retrieved_items)} 条相似提示词")
            
            # 2. 生成阶段（流式展示）
            status_text.text("✨ 步骤 2/2: 正在调用 Ollama 生成 Prompt（流式输出）...")
            progress_bar.progress(40)
            
            # 构建上下文
            context = st.session_state.rag_generator._build_context(user_input, retrieved_items)
            user_prompt = f"{context}\n\n请根据以上信息，生成一段高质量的中文绘图提示词："
            
            generate_start = time.time()
            token_placeholder = st.empty()
            token_buffer = []
            first_token_time = None
            
            for tok in st.session_state.rag_generator.client.stream_generate(
                prompt=user_prompt,
                system=st.session_state.rag_generator.system_prompt,
                temperature=0.7
            ):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - generate_start
                    status_text.text(f"✨ 已收到首个 token，TTFT: {ttft:.3f} 秒")
                    progress_bar.progress(70)
                token_buffer.append(tok)
                token_placeholder.text("".join(token_buffer))
            
            generate_time = time.time() - generate_start
            final_prompt = "".join(token_buffer).strip()
            
            progress_bar.progress(100)
            status_text.text(f"✓ 生成完成！总耗时: {search_time + generate_time:.3f}秒")
            
            # 组装结果
            result = {
                "final_prompt": final_prompt,
                "references": retrieved_items,
                "user_intent": user_input
            }
            
            # 清除进度条
            progress_bar.empty()
            status_text.empty()
            
            # 显示结果
            st.markdown("---")
            st.subheader("✨ 生成的中文 Prompt")
            
            # 显示性能统计
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                st.metric("🔍 检索耗时", f"{search_time:.3f}秒")
            with col_perf2:
                st.metric("✨ 生成耗时", f"{generate_time:.3f}秒")
            with col_perf3:
                st.metric("⏱️ 总耗时", f"{search_time + generate_time:.3f}秒")
            if first_token_time:
                st.caption(f"TTFT (首 token 延迟): {ttft:.3f} 秒")
            
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
                        elements = ref.get('visual_elements', [])
                        tech = ref.get('technical', [])
                        st.markdown(f"**视觉元素**: {', '.join(elements[:5]) if elements else 'N/A'}")
                        st.markdown(f"**技术参数**: {', '.join(tech[:5]) if tech else 'N/A'}")
                    
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

