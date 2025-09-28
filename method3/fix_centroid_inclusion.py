#!/usr/bin/env python3
"""
修改tune_kmeans_hnsw_optimized.py中的两个_stage2_child_search方法，
将include_centroids_in_results的条件判断改为默认包含质心
"""

def fix_centroid_inclusion():
    file_path = 'tune_kmeans_hnsw_optimized.py'
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换两个地方的条件判断
    old_pattern = """        # 可选择性地包含质心在结果中
        if self.include_centroids_in_results:
            for centroid_id, _ in closest_centroids:
                # 将质心作为候选节点（使用质心向量）
                centroid_idx = self.centroid_ids.index(centroid_id)
                centroid_vector = self.centroids[centroid_idx]
                # 临时存储质心向量
                self.child_vectors[centroid_id] = centroid_vector
                candidate_children.add(centroid_id)"""
    
    new_pattern = """        # 默认包含质心在结果中
        for centroid_id, _ in closest_centroids:
            # 将质心作为候选节点（使用质心向量）
            centroid_idx = self.centroid_ids.index(centroid_id)
            centroid_vector = self.centroids[centroid_idx]
            # 临时存储质心向量
            self.child_vectors[centroid_id] = centroid_vector
            candidate_children.add(centroid_id)"""
    
    # 替换所有匹配项
    new_content = content.replace(old_pattern, new_pattern)
    
    if new_content != content:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Successfully modified include_centroids_in_results logic!")
        
        # 统计替换次数
        count = content.count(old_pattern)
        print(f"Replaced {count} occurrences")
    else:
        print("No modifications needed or no target found.")

if __name__ == '__main__':
    fix_centroid_inclusion()
