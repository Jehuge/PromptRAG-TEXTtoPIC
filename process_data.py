"""
数据处理脚本：从 Excel/CSV 处理数据并生成结构化 JSONL
"""
import sys
import os
from etl_pipeline import ETLPipeline
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    """主函数"""
    print("="*60)
    print("数据处理工具 (ETL Pipeline)")
    print("="*60)
    
    # 查找数据文件
    data_files = []
    if os.path.exists(RAW_DATA_DIR):
        for file in os.listdir(RAW_DATA_DIR):
            if file.endswith(('.xlsx', '.xls', '.csv')):
                data_files.append(os.path.join(RAW_DATA_DIR, file))
    
    if not data_files:
        print(f"\n✗ 未找到数据文件，请将 Excel/CSV 文件放入以下目录:")
        print(f"  {RAW_DATA_DIR}")
        print(f"\n正在创建目录...")
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        return
    
    # 选择文件
    if len(data_files) == 1:
        selected_file = data_files[0]
        print(f"\n找到文件: {selected_file}")
    else:
        print("\n找到多个数据文件:")
        for i, f in enumerate(data_files, 1):
            print(f"  {i}. {f}")
        
        choice = input("\n请选择文件编号 (直接回车使用第一个): ").strip()
        if choice:
            try:
                selected_file = data_files[int(choice) - 1]
            except (ValueError, IndexError):
                print("无效选择，使用第一个文件")
                selected_file = data_files[0]
        else:
            selected_file = data_files[0]
    
    # 测试连接
    print("\n正在测试 Ollama 连接...")
    pipeline = ETLPipeline()
    if not pipeline.client.test_connection():
        print("\n✗ 无法连接到 Ollama 服务，请检查:")
        print("  1. PC 端 Ollama 是否运行")
        print("  2. 网络连接是否正常")
        print("  3. config.py 中的 OLLAMA_HOST 配置是否正确")
        return
    
    # 加载数据
    print(f"\n正在加载数据: {selected_file}")
    if selected_file.endswith('.csv'):
        texts = pipeline.load_csv(selected_file)
    else:
        texts = pipeline.load_excel(selected_file)
    
    if not texts:
        print("✗ 未加载到任何数据")
        return
    
    # 确认处理
    print(f"\n将处理 {len(texts)} 条记录")
    print("注意: 这可能需要较长时间，请耐心等待...")
    confirm = input("\n确认开始处理？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # 处理数据
    try:
        output_path = pipeline.process_batch(texts)
        print(f"\n✓ 处理完成！输出文件: {output_path}")
        print("\n下一步: 运行 python build_index.py 构建向量索引")
    except Exception as e:
        print(f"\n✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

