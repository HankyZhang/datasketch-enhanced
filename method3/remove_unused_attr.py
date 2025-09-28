#!/usr/bin/env python3
"""
删除tune_kmeans_hnsw_optimized.py中不再需要的include_centroids_in_results属性设置
"""

def remove_unused_attribute():
    file_path = 'tune_kmeans_hnsw_optimized.py'
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤掉包含include_centroids_in_results设置的行
    new_lines = []
    skip_next = False
    
    for line in lines:
        if 'include_centroids_in_results设置' in line:
            skip_next = True
            continue
        elif skip_next and 'self.include_centroids_in_results' in line:
            skip_next = False
            continue
        else:
            new_lines.append(line)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("Successfully removed unused include_centroids_in_results attributes!")

if __name__ == '__main__':
    remove_unused_attribute()
