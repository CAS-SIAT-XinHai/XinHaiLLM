import os
import json
import sys
import math


def calculate_average_scores(uuid_folder):
    # 定义四个分类
    categories = {
        "KG-单选": [],
        "KG-多选": [],
        "CA-单选": [],
        "CA-多选": []
    }

    # 遍历文件夹，找到所有 JSON 结果文件
    for file_name in os.listdir(uuid_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(uuid_folder, file_name)

            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                accuracy = data.get('accuracy', None)

                if accuracy is None or math.isnan(accuracy):
                    print(f"Warning: No valid accuracy found in {file_name}, skipping.")
                    continue

                # 打印调试信息，查看文件名和 accuracy 值
                print(f"Processing {file_name}, accuracy: {accuracy}")

                # 根据文件名分类
                if "KG" in file_name and "单项选择题" in file_name:
                    categories["KG-单选"].append(accuracy)
                elif "KG" in file_name and "多项选择题" in file_name:
                    categories["KG-多选"].append(accuracy)
                elif "CA" in file_name and "单项选择题" in file_name:
                    categories["CA-单选"].append(accuracy)
                elif "CA" in file_name and "多项选择题" in file_name:
                    categories["CA-多选"].append(accuracy)

    # 计算并输出每个类别的平均分
    for category, scores in categories.items():
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            print(f"{category} 平均分: {avg_score:.2f}%")
        else:
            print(f"{category} 没有数据")


if __name__ == "__main__":
    # 获取 UUID 文件夹路径
    if len(sys.argv) != 2:
        print("Usage: python calculate_avg_scores.py <uuid_folder>")
        sys.exit(1)

    uuid_folder = sys.argv[1]

    # 检查路径是否存在
    if not os.path.exists(uuid_folder):
        print(f"Error: Folder '{uuid_folder}' does not exist.")
        sys.exit(1)

    # 计算并打印平均分
    calculate_average_scores(uuid_folder)
