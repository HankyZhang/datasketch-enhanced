#!/usr/bin/env python3
"""
修复优化版本中的质心包含逻辑，使其与原始版本保持一致
"""

def fix_stage2_search_logic():
    file_path = "c:/Code/datasketch-enhanced/method3/tune_kmeans_hnsw_optimized.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到并修复两个_stage2_child_search方法
    modified = False
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 寻找_stage2_child_search方法的开始
        if "def _stage2_child_search" in line:
            print(f"Found _stage2_child_search at line {i+1}")
            
            # 向前找到质心包含逻辑
            j = i + 1
            while j < len(lines) and j < i + 20:  # 在方法开始后的20行内查找
                if "# 默认包含质心在结果中" in lines[j]:
                    print(f"  Found centroid inclusion logic at line {j+1}")
                    
                    # 修改注释
                    lines[j] = lines[j].replace("# 默认包含质心在结果中", "# 条件性包含质心在结果中（与原始版本保持一致）")
                    
                    # 在下一行添加条件判断
                    if j+1 < len(lines) and "for centroid_id, _ in closest_centroids:" in lines[j+1]:
                        lines[j+1] = "        if self.include_centroids_in_results:\n" + "    " + lines[j+1]
                        modified = True
                        print(f"  Modified centroid inclusion logic")
                    break
                j += 1
        i += 1
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Successfully fixed stage2 search logic")
    else:
        print("No modifications made")

if __name__ == "__main__":
    fix_stage2_search_logic()
