import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="生成量化数据集")
    parser.add_argument("--txt", type=str, default="dataset_list.txt", help="输出的txt文件路径")
    parser.add_argument("--output_dir", type=str, default="datasets", help="存储npy文件的文件夹路径")
    parser.add_argument("--channel", type=int, default=3, help="通道数")
    parser.add_argument("--height", type=int, default=224, help="高度")
    parser.add_argument("--width", type=int, default=224, help="宽度")
    parser.add_argument("--num", type=int, default=10, help="生成的数据集数量")
    args = parser.parse_args()

    # 获取脚本所在目录作为基准路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理输出目录路径
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(base_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    # 处理txt文件路径
    if not os.path.isabs(args.txt):
        txt_path = os.path.join(base_dir, args.txt)
    else:
        txt_path = args.txt

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成npy文件并记录相对路径
    npy_files = []
    for i in range(args.num):
        # 生成随机数据 
        data = np.random.randint(0,256, (1, args.channel, args.height, args.width), dtype=np.uint8)
        
        # 保存npy文件
        filename = f"data_{i:04d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, data)
        
        # 记录相对路径
        rel_path = os.path.relpath(filepath, base_dir)
        npy_files.append(rel_path)
        
        print(f"生成: {rel_path}")

    # 写入txt文件
    with open(txt_path, "w") as f:
        for npy_file in npy_files:
            f.write(npy_file + "\n")

    print(f"\n完成!")
    print(f"数据形状: [1, {args.channel}, {args.height}, {args.width}]")
    print(f"生成数量: {args.num}")
    print(f"存储目录: {output_dir}")
    print(f"列表文件: {txt_path}")


if __name__ == "__main__":
    main()
