import streamlit as st
import sys
import platform

# デバッグ情報
st.sidebar.write(f"Python: {platform.python_version()}")
st.sidebar.write(f"Platform: {platform.system()}")

# OpenCVのインポートを試行
try:
    import cv2
    st.sidebar.success(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {e}")
    st.error("Please check requirements.txt and packages.txt")
    st.stop()

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import collections
import csv
import io
from PIL import Image
import tempfile
import os

# ページ設定
st.set_page_config(
    page_title="画像グラフ生成アプリ",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("📊 画像からグラフデータを生成")
st.markdown("画像をアップロードして、スケルトン化とグラフ構築を行い、CSVデータを生成します。")

# サイドバーでパラメータ設定
st.sidebar.header("⚙️ 設定")

# 画像処理設定
st.sidebar.subheader("画像処理")
resize_enabled = st.sidebar.checkbox("画像を480x360にリサイズ", value=True)
crop_bottom = st.sidebar.slider("下部トリミング (ピクセル)", 0, 100, 50)

# グラフ構築設定
st.sidebar.subheader("グラフ構築")
curvature_threshold = st.sidebar.slider("曲率分割閾値", 1.0, 20.0, 10.0, 0.5)
max_jump_distance = st.sidebar.slider("最大ジャンプ距離", 1, 5, 2)
min_intersection_transitions = st.sidebar.slider("交差点検出閾値", 2, 5, 3)
min_node_area = st.sidebar.slider("最小ノード面積", 1, 10, 1)

# ファイルアップロード
uploaded_file = st.file_uploader("画像ファイルをアップロード", type=['png', 'jpg', 'jpeg'])


# --- 関数定義（元のコードから抽出） ---

def resize_image(img, target_width=480, target_height=360):
    """画像をリサイズ"""
    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_img, original_height, original_width


def crop_bottom_image(img, crop_pixels):
    """画像下部をトリミング"""
    if crop_pixels <= 0:
        return img, img.shape[0]
    
    height, width = img.shape[:2]
    new_height = height - crop_pixels
    
    if new_height <= 0:
        return None, 0
    
    cropped_img = img[0:new_height, :]
    return cropped_img, new_height


def refine_skeleton_branches(skeleton):
    """スケルトンの分岐を整理"""
    H, W = skeleton.shape
    refined = skeleton.copy()
    
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                   (1, 1), (1, 0), (1, -1), (0, -1)]
    
    endpoints = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skeleton[y, x] == 1:
                neighbors = [skeleton[y + dy, x + dx] for dy, dx in neighbors_8]
                neighbor_count = sum(neighbors)
                
                if neighbor_count == 1:
                    endpoints.append((y, x))
    
    min_branch_length = 5
    for start_y, start_x in endpoints:
        branch_length = 0
        y, x = start_y, start_x
        branch_pixels = [(y, x)]
        
        while branch_length < min_branch_length:
            neighbors = []
            for dy, dx in neighbors_8:
                ny, nx = y + dy, x + dx
                if (0 <= ny < H and 0 <= nx < W and 
                    skeleton[ny, nx] == 1 and (ny, nx) not in branch_pixels):
                    neighbors.append((ny, nx))
            
            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                y, x = neighbors[0]
                branch_pixels.append((y, x))
                branch_length += 1
            else:
                break
        
        if branch_length < min_branch_length and len(neighbors) == 0:
            for py, px in branch_pixels:
                refined[py, px] = 0
    
    return refined


def high_quality_skeletonization(img):
    """高品質スケルトン化"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=11, 
        C=2
    )
    
    kernel_small = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(cleaned, kernel_dilate, iterations=1)
    
    binary_bool = (dilated > 128).astype(bool)
    skeleton_bool = skeletonize(binary_bool)
    skeleton = skeleton_bool.astype(np.uint8)
    
    labeled_skeleton = label(skeleton, connectivity=2)
    regions = regionprops(labeled_skeleton)
    
    min_component_size = 5
    filtered_skeleton = np.zeros_like(skeleton)
    for region in regions:
        if region.area >= min_component_size:
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    
    filtered_skeleton = refine_skeleton_branches(filtered_skeleton)
    
    processed_img = (filtered_skeleton * 255).astype(np.uint8)
    
    return filtered_skeleton, processed_img


def detect_and_build_graph(binary_img, curvature_threshold, max_jump, min_transitions, min_area):
    """グラフ検出と構築"""
    H, W = binary_img.shape
    
    feature_map = np.zeros_like(binary_img)
    neighbors_coord = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    feature_pixels = {}
    
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if binary_img[y, x] == 1:
                neighbors = [(binary_img[y + dy, x + dx]) for dy, dx in neighbors_coord]
                transitions = sum(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1 for i in range(8))
                
                is_feature = False
                node_type = -1
                
                if transitions >= min_transitions:
                    is_feature = True
                    node_type = 0
                elif transitions == 1:
                    is_feature = True
                    node_type = 2
                elif transitions == 2:
                    white_indices = [i for i, val in enumerate(neighbors) if val]
                    if len(white_indices) == 2:
                        idx1, idx2 = white_indices
                        distance = min(abs(idx1 - idx2), 8 - abs(idx1 - idx2))
                        if distance == 2:
                            is_feature = True
                            node_type = 1
                
                if is_feature:
                    feature_map[y, x] = 1
                    feature_pixels[(y, x)] = node_type
    
    if feature_map.sum() == 0:
        return None, None, None
    
    labeled_img = label(feature_map, connectivity=2)
    regions = regionprops(labeled_img)
    
    nodes = {}
    coord_to_node_id = np.full((H, W), -1, dtype=int)
    node_id_counter = 1
    
    for region in regions:
        if region.area < min_area:
            continue
        node_id = node_id_counter
        center_y, center_x = region.centroid
        
        cluster_types = [feature_pixels[(py, px)] for py, px in region.coords if (py, px) in feature_pixels]
        if cluster_types:
            most_common_type = collections.Counter(cluster_types).most_common(1)[0][0]
        else:
            continue
        
        nodes[node_id] = {
            'pos': (int(center_x), int(center_y)), 
            'type': most_common_type, 
            'adj': [], 
            'coords': region.coords
        }
        for y, x in region.coords:
            coord_to_node_id[y, x] = node_id
        node_id_counter += 1
    
    if len(nodes) == 0:
        return None, None, None
    
    marked_img = cv2.cvtColor(binary_img * 255, cv2.COLOR_GRAY2BGR)
    edges = set()
    edge_visited_map = np.full((H, W), -1, dtype=int)
    edge_id_counter = 0
    
    start_pixels = []
    for node_id, node_data in nodes.items():
        for start_y, start_x in node_data['coords']:
            for dy, dx in neighbors_coord:
                neighbor_y, neighbor_x = start_y + dy, start_x + dx
                if (0 <= neighbor_y < H and 0 <= neighbor_x < W and 
                    binary_img[neighbor_y, neighbor_x] == 1 and 
                    coord_to_node_id[neighbor_y, neighbor_x] == -1):
                    start_pixels.append((node_id, start_y, start_x, neighbor_y, neighbor_x))
    
    processed_starts = set()
    
    for node_id, start_y, start_x, initial_y, initial_x in start_pixels:
        start_key = (node_id, initial_y, initial_x)
        if start_key in processed_starts:
            continue
        if edge_visited_map[initial_y, initial_x] != -1:
            continue
        
        path = []
        temp_path_visited = set()
        y, x = initial_y, initial_x
        prev_dy, prev_dx = initial_y - start_y, initial_x - start_x
        current_curvature = 0.0
        current_start_node_id = node_id
        
        while True:
            end_node_id_check = coord_to_node_id[y, x]
            is_end_node = (end_node_id_check != -1 and end_node_id_check != current_start_node_id)
            is_split_point = (current_curvature >= curvature_threshold) and (end_node_id_check == -1)
            
            if is_end_node or is_split_point:
                target_node_id = -1
                if is_end_node:
                    target_node_id = end_node_id_check
                elif is_split_point:
                    target_node_id = node_id_counter
                    nodes[target_node_id] = {
                        'pos': (x, y), 
                        'type': 3, 
                        'adj': [], 
                        'coords': [(y, x)]
                    }
                    coord_to_node_id[y, x] = target_node_id
                    node_id_counter += 1
                
                n1, n2 = min(current_start_node_id, target_node_id), max(current_start_node_id, target_node_id)
                edge_key = (n1, n2)
                
                if current_start_node_id == node_id or edge_key not in edges:
                    edges.add(edge_key)
                    length = len(path)
                    nodes[current_start_node_id]['adj'].append((target_node_id, length))
                    nodes[target_node_id]['adj'].append((current_start_node_id, length))
                    
                    edge_id_counter += 1
                    for py, px in path:
                        marked_img[py, px] = (0, 255, 0)
                        edge_visited_map[py, px] = edge_id_counter
                
                if is_end_node:
                    break
                elif is_split_point:
                    current_start_node_id = target_node_id
                    current_curvature = 0.0
                    path = []
            
            if edge_visited_map[y, x] != -1:
                break
            
            path.append((y, x))
            temp_path_visited.add((y, x))
            
            best_pixel = None
            best_vector = (0, 0)
            best_score = -2
            
            for dy_search in range(-max_jump, max_jump + 1):
                for dx_search in range(-max_jump, max_jump + 1):
                    if dy_search == 0 and dx_search == 0:
                        continue
                    next_y, next_x = y + dy_search, x + dx_search
                    
                    if not (0 <= next_y < H and 0 <= next_x < W):
                        continue
                    if (next_y, next_x) in temp_path_visited or edge_visited_map[next_y, next_x] != -1:
                        continue
                    
                    if binary_img[next_y, next_x] == 1:
                        current_vector = (dy_search, dx_search)
                        is_adjacent = max(abs(dy_search), abs(dx_search)) == 1
                        is_jump = max(abs(dy_search), abs(dx_search)) == 2
                        
                        if is_adjacent:
                            score = prev_dy * dy_search + prev_dx * dx_search
                            if score > best_score:
                                best_score = score
                                best_pixel = (next_y, next_x)
                                best_vector = current_vector
                        
                        elif is_jump:
                            mid_y1, mid_x1 = y + dy_search//2, x + dx_search//2
                            if binary_img[mid_y1, mid_x1] == 0:
                                score = prev_dy * dy_search + prev_dx * dx_search - 3
                                if score > best_score:
                                    best_score = score
                                    best_pixel = (next_y, next_x)
                                    best_vector = current_vector
            
            if best_pixel:
                new_dy, new_dx = best_vector
                curvature_change = 2 - (prev_dy * new_dy + prev_dx * new_dx)
                current_curvature += curvature_change
                
                if max(abs(best_vector[0]), abs(best_vector[1])) == 2:
                    mid_y, mid_x = y + best_vector[0]//2, x + best_vector[1]//2
                    path.append((mid_y, mid_x))
                    temp_path_visited.add((mid_y, mid_x))
                
                y, x = best_pixel
                prev_dy, prev_dx = best_vector
            else:
                break
        
        processed_starts.add((node_id, initial_y, initial_x))
    
    # ノードを描画
    for node_id, data in nodes.items():
        x, y = data['pos']
        if data['type'] == 0:
            color = (255, 0, 0)  # 交差点
        elif data['type'] == 1:
            color = (0, 0, 255)  # カーブ
        elif data['type'] == 2:
            color = (0, 255, 255)  # 端点
        elif data['type'] == 3:
            color = (0, 165, 255)  # 曲率分割
        
        radius = 5 if data['type'] != 3 else 3
        cv2.circle(marked_img, (x, y), radius, color, -1)
    
    return nodes, edges, marked_img


def create_csv_data(nodes, edges, image_height):
    """CSVデータを作成"""
    type_labels = {
        0: 'Intersection',
        1: 'Curve/Corner (Topology)',
        2: 'Endpoint',
        3: 'Intermediate (Curvature Split)'
    }
    
    # ノードCSV
    node_data = []
    for node_id, data in nodes.items():
        x_pixel, y_pixel = data['pos']
        node_type = data['type']
        
        x_scratch = int(round(x_pixel - 240))
        y_scratch = int(round(image_height / 2 - y_pixel))
        
        node_data.append([
            node_id,
            x_scratch,
            y_scratch,
            node_type,
            type_labels.get(node_type, 'Unknown')
        ])
    
    # エッジCSV
    edge_data = []
    edge_id_counter = 1
    unique_edges = set()
    
    for node_id, data in nodes.items():
        for neighbor_id, length in data['adj']:
            n1, n2 = min(node_id, neighbor_id), max(node_id, neighbor_id)
            edge_key = (n1, n2)
            
            if edge_key not in unique_edges:
                unique_edges.add(edge_key)
                edge_data.append([edge_id_counter, n1, n2, length])
                edge_id_counter += 1
    
    return node_data, edge_data


def create_csv_file(data, header):
    """CSV文字列を作成"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()


# --- メイン処理 ---

if uploaded_file is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.success("✅ 画像をアップロードしました")
    
    # 処理実行ボタン
    if st.button("🚀 グラフデータを生成", type="primary"):
        with st.spinner("処理中..."):
            progress_bar = st.progress(0)
            
            # ステップ1: リサイズ
            if resize_enabled:
                st.info("ステップ 1/4: 画像リサイズ中...")
                img, orig_h, orig_w = resize_image(img, 480, 360)
                current_height = 360
                progress_bar.progress(25)
            else:
                current_height = img.shape[0]
            
            # ステップ2: トリミング
            st.info("ステップ 2/4: トリミング中...")
            if crop_bottom > 0:
                img, current_height = crop_bottom_image(img, crop_bottom)
                if img is None:
                    st.error("❌ トリミングに失敗しました")
                    st.stop()
            progress_bar.progress(40)
            
            # ステップ3: スケルトン化
            st.info("ステップ 3/4: スケルトン化中...")
            skeleton_data, skeleton_visual = high_quality_skeletonization(img)
            progress_bar.progress(60)
            
            # ステップ4: グラフ構築
            st.info("ステップ 4/4: グラフ構築中...")
            nodes_data, edges_set, marked_img = detect_and_build_graph(
                skeleton_data,
                curvature_threshold,
                max_jump_distance,
                min_intersection_transitions,
                min_node_area
            )
            progress_bar.progress(100)
            
            if nodes_data is None or edges_set is None:
                st.error("❌ グラフの検出に失敗しました。パラメータを調整してください。")
            else:
                st.success(f"✅ 処理完了! ノード数: {len(nodes_data)}, エッジ数: {len(edges_set)}")
                
                # 結果表示
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("元画像")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("スケルトン画像")
                    st.image(skeleton_visual, use_container_width=True)
                
                with col3:
                    st.subheader("グラフ画像")
                    st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # CSVデータ生成
                node_data, edge_data = create_csv_data(nodes_data, edges_set, current_height)
                
                # ダウンロードボタン
                st.subheader("📥 データダウンロード")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    node_csv = create_csv_file(
                        node_data,
                        ['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label']
                    )
                    st.download_button(
                        label="ノードCSVをダウンロード",
                        data=node_csv,
                        file_name="nodes.csv",
                        mime="text/csv"
                    )
                
                with col_dl2:
                    edge_csv = create_csv_file(
                        edge_data,
                        ['edge_id', 'from_node_id', 'to_node_id', 'pixel_length']
                    )
                    st.download_button(
                        label="エッジCSVをダウンロード",
                        data=edge_csv,
                        file_name="edges.csv",
                        mime="text/csv"
                    )
                
                with col_dl3:
                    # グラフ画像をダウンロード
                    is_success, buffer = cv2.imencode(".png", marked_img)
                    if is_success:
                        st.download_button(
                            label="グラフ画像をダウンロード",
                            data=buffer.tobytes(),
                            file_name="graph_marked.png",
                            mime="image/png"
                        )
                
                # データプレビュー
                with st.expander("📊 ノードデータプレビュー"):
                    st.write(f"総ノード数: {len(node_data)}")
                    import pandas as pd
                    df_nodes = pd.DataFrame(
                        node_data,
                        columns=['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label']
                    )
                    st.dataframe(df_nodes.head(10))
                
                with st.expander("🔗 エッジデータプレビュー"):
                    st.write(f"総エッジ数: {len(edge_data)}")
                    df_edges = pd.DataFrame(
                        edge_data,
                        columns=['edge_id', 'from_node_id', 'to_node_id', 'pixel_length']
                    )
                    st.dataframe(df_edges.head(10))

else:
    st.info("👆 左側のファイルアップローダーから画像を選択してください")
    
    # 使い方の説明
    with st.expander("📖 使い方"):
        st.markdown("""
        ### 使い方
        
        1. **画像をアップロード**: 左のサイドバーから画像ファイルを選択
        2. **パラメータ調整**: サイドバーで各種パラメータを調整
        3. **生成開始**: 「グラフデータを生成」ボタンをクリック
        4. **結果確認**: 生成されたグラフとデータを確認
        5. **ダウンロード**: CSVファイルと画像をダウンロード
        
        ### パラメータ説明
        
        - **画像リサイズ**: 処理速度向上のため480x360にリサイズ
        - **下部トリミング**: 不要な下部領域を削除
        - **曲率分割閾値**: 大きいほど直線として認識しやすい
        - **最大ジャンプ距離**: ノイズ耐性（通常は2推奨）
        - **交差点検出閾値**: 交差点判定の感度
        - **最小ノード面積**: 小さなノイズを除去
        """)
    
    # カラー凡例
    with st.expander("🎨 ノードの色の意味"):
        col_legend1, col_legend2, col_legend3, col_legend4 = st.columns(4)
        
        with col_legend1:
            st.markdown("🔴 **赤**: 交差点")
        with col_legend2:
            st.markdown("🔵 **青**: カーブ/コーナー")
        with col_legend3:
            st.markdown("🟡 **黄**: 端点")
        with col_legend4:
            st.markdown("🟠 **オレンジ**: 曲率分割点")

# フッター
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
