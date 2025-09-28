#!/usr/bin/env python3
"""
修复tune_kmeans_hnsw_optimized.py中两个_stage2_child_search方法，
为它们添加include_centroids_in_results功能
"""

def fix_stage2_search():
    file_path = 'tune_kmeans_hnsw_optimized.py'
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到需要修改的位置
    # 第一个_stage2_child_search方法在单枢纽系统中（大约第361行）
    # 第二个在多枢纽系统中（大约第777行）
    
    modified = False
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 查找_stage2_child_search方法定义
        if 'def _stage2_child_search(' in line:
            print(f"Found _stage2_child_search method at line {i+1}")
            
            # 找到"if not candidate_children:"这一行
            j = i
            while j < len(lines) and 'if not candidate_children:' not in lines[j]:
                j += 1
            
            if j < len(lines):
                print(f"Found 'if not candidate_children:' at line {j+1}")
                
                # 在这一行之前插入include_centroids_in_results逻辑
                indent = '        '  # 方法内部的缩进
                centroid_logic = [
                    f"{indent}# 可选择性地包含质心在结果中\n",
                    f"{indent}if self.include_centroids_in_results:\n",
                    f"{indent}    for centroid_id, _ in closest_centroids:\n",
                    f"{indent}        # 将质心作为候选节点（使用质心向量）\n",
                    f"{indent}        centroid_idx = self.centroid_ids.index(centroid_id)\n",
                    f"{indent}        centroid_vector = self.centroids[centroid_idx]\n",
                    f"{indent}        # 临时存储质心向量\n",
                    f"{indent}        self.child_vectors[centroid_id] = centroid_vector\n",
                    f"{indent}        candidate_children.add(centroid_id)\n",
                    f"{indent}\n"
                ]
                
                # 插入新的逻辑
                lines[j:j] = centroid_logic
                modified = True
                
                # 跳过这个方法的其余部分以避免重复处理
                i = j + len(centroid_logic) + 10
            else:
                i += 1
        else:
            i += 1
    
    if modified:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("File successfully modified!")
    else:
        print("No modifications needed or no target found.")

if __name__ == '__main__':
    fix_stage2_search()
