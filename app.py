import streamlit as st
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import collections
import csv
import io
from PIL import Image
import tempfile
import os
import math

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

# 距離スケール設定
st.sidebar.subheader("📏 距離スケール設定")
enable_distance_scale = st.sidebar.checkbox("実距離計算を有効化", value=False)

if enable_distance_scale:
    st.sidebar.markdown("**画像の範囲（緯度経度）**")
    
    col_lat1, col_lat2 = st.sidebar.columns(2)
    with col_lat1:
        north_latitude = st.number_input("北緯度", value=35.1, format="%.6f", step=0.000001)
    with col_lat2:
        south_latitude = st.number_input("南緯度", value=35.0, format="%.6f", step=0.000001)
    
    col_lon1, col_lon2 = st.sidebar.columns(2)
    with col_lon1:
        west_longitude = st.number_input("西経度", value=135.0, format="%.6f", step=0.000001)
    with col_lon2:
        east_longitude = st.number_input("東経度", value=135.1, format="%.6f", step=0.000001)
    
    # 画像サイズ（ピクセル）
    st.sidebar.markdown("**画像サイズ（ピクセル）**")
    col_size1, col_size2 = st.sidebar.columns(2)
    with col_size1:
        image_width_px = st.number_input("幅", value=480, min_value=1)
    with col_size2:
        image_height_px = st.number_input("高さ", value=360, min_value=1)

# 画像処理設定
st.sidebar.subheader("画像処理")
resize_enabled = st.sidebar.checkbox("画像を480x360にリサイズ", value=True)

# グラフ構築設定
st.sidebar.subheader("グラフ構築")
curvature_threshold = st.sidebar.slider("曲率分割閾値", 1.0, 20.0, 10.0, 0.5)
max_jump_distance = st.sidebar.slider("最大ジャンプ距離", 1, 5, 2)
min_intersection_transitions = st.sidebar.slider("交差点検出閾値", 2, 5, 3)
min_node_area = st.sidebar.slider("最小ノード面積", 1, 10, 1)

# ファイルアップロード
uploaded_file = st.file_uploader("画像ファイルをアップロード", type=['png', 'jpg', 'jpeg'])


# --- 関数定義 ---

def calculate_distance_scale(north_lat, south_lat, west_lon, east_lon, width_px, height_px):
    EARTH_RADIUS = 6371000
    center_lat = (north_lat + south_lat) / 2
    center_lat_rad = math.radians(center_lat)
    lon_diff = abs(east_lon - west_lon)
    lon_diff_rad = math.radians(lon_diff)
    distance_x_meters = EARTH_RADIUS * lon_diff_rad * math.cos(center_lat_rad)
    meters_per_pixel_x = distance_x_meters / width_px
    lat_diff = abs(north_lat - south_lat)
    lat_diff_rad = math.radians(lat_diff)
    distance_y_meters = EARTH_RADIUS * lat_diff_rad
    meters_per_pixel_y = distance_y_meters / height_px
    meters_per_pixel_avg = (meters_per_pixel_x + meters_per_pixel_y) / 2
    return meters_per_pixel_x, meters_per_pixel_y, meters_per_pixel_avg


def resize_image(img, target_width=480, target_height=360):
    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_img, original_height, original_width


def refine_skeleton_branches(skeleton):
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
    H, W = binary_img.shape
    feature_map = np.zeros_like(binary_img)
    neighbors_coord = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                       (1, 1), (1, 0), (1, -1), (0, -1)]
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
                            mid_y1, mid_x1 = y + dy_search // 2, x + dx_search // 2
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
                    mid_y, mid_x = y + best_vector[0] // 2, x + best_vector[1] // 2
                    path.append((mid_y, mid_x))
                    temp_path_visited.add((mid_y, mid_x))
                y, x = best_pixel
                prev_dy, prev_dx = best_vector
            else:
                break

        processed_starts.add((node_id, initial_y, initial_x))

    # ノード描画
    for node_id, data in nodes.items():
        x, y = data['pos']
        if data['type'] == 0:
            color = (255, 0, 0)
        elif data['type'] == 1:
            color = (0, 0, 255)
        elif data['type'] == 2:
            color = (0, 255, 255)
        elif data['type'] == 3:
            color = (0, 165, 255)
        radius = 5 if data['type'] != 3 else 3
        cv2.circle(marked_img, (x, y), radius, color, -1)

    return nodes, edges, marked_img


def create_csv_data(nodes, edges, image_height, meters_per_pixel=None):
    type_labels = {
        0: 'Intersection',
        1: 'Curve/Corner (Topology)',
        2: 'Endpoint',
        3: 'Intermediate (Curvature Split)'
    }
    node_data = []
    for node_id, data in nodes.items():
        x_pixel, y_pixel = data['pos']
        node_type = data['type']
        x_scratch = int(round(x_pixel - 240))
        y_scratch = int(round(image_height / 2 - y_pixel))
        node_data.append([node_id, x_scratch, y_scratch, node_type, type_labels.get(node_type, 'Unknown')])

    edge_data = []
    edge_id_counter = 1
    unique_edges = set()
    for node_id, data in nodes.items():
        for neighbor_id, length in data['adj']:
            n1, n2 = min(node_id, neighbor_id), max(node_id, neighbor_id)
            edge_key = (n1, n2)
            if edge_key not in unique_edges:
                unique_edges.add(edge_key)
                if meters_per_pixel is not None:
                    distance_meters = length * meters_per_pixel
                    edge_data.append([edge_id_counter, n1, n2, length, f"{distance_meters:.2f}"])
                else:
                    edge_data.append([edge_id_counter, n1, n2, length])
                edge_id_counter += 1
    return node_data, edge_data


def create_csv_file(data, header):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()


# --- メイン処理 ---
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.success("✅ 画像をアップロードしました")

    if enable_distance_scale:
        m_per_px_x, m_per_px_y, m_per_px_avg = calculate_distance_scale(
            north_latitude, south_latitude, west_longitude, east_longitude, image_width_px, image_height_px)
        center_lat = (north_latitude + south_latitude) / 2
        st.info(f"📏 **距離スケール計算結果** (中心緯度: {center_lat:.6f}°)\n\n"
                f"- 横方向: 1px = {m_per_px_x:.2f} m\n"
                f"- 縦方向: 1px = {m_per_px_y:.2f} m\n"
                f"- 平均: 1px = {m_per_px_avg:.2f} m")

    if st.button("🚀 グラフデータを生成", type="primary"):
        with st.spinner("処理中..."):
            progress_bar = st.progress(0)
            if resize_enabled:
                st.info("ステップ 1/3: 画像リサイズ中...")
                img, orig_h, orig_w = resize_image(img, 480, 360)
                current_height = 360
                progress_bar.progress(25)
            else:
                current_height = img.shape[0]

            st.info("ステップ 2/3: スケルトン化中...")
            skeleton_data, skeleton_visual = high_quality_skeletonization(img)
            progress_bar.progress(60)

            st.info("ステップ 3/3: グラフ構築中...")
            nodes_data, edges_set, marked_img = detect_and_build_graph(
                skeleton_data, curvature_threshold, max_jump_distance,
                min_intersection_transitions, min_node_area)
            progress_bar.progress(100)

            if nodes_data is None or edges_set is None:
                st.error("❌ グラフの検出に失敗しました。パラメータを調整してください。")
            else:
                st.success(f"✅ 完了! ノード数: {len(nodes_data)}, エッジ数: {len(edges_set)}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("元画像")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                with col2:
                    st.subheader("スケルトン画像")
                    st.image(skeleton_visual)
                with col3:
                    st.subheader("グラフ画像")
                    st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))

                if enable_distance_scale:
                    node_data, edge_data = create_csv_data(nodes_data, edges_set, current_height, m_per_px_avg)
                else:
                    node_data, edge_data = create_csv_data(nodes_data, edges_set, current_height)

                st.subheader("📥 データダウンロード")
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    node_csv = create_csv_file(node_data, ['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label'])
                    st.download_button("ノードCSVをダウンロード", node_csv, "nodes.csv", "text/csv")

                with col_dl2:
                    edge_header = ['edge_id', 'from_node_id', 'to_node_id', 'pixel_length']
                    if enable_distance_scale:
                        edge_header.append('distance_meters')
                    edge_csv = create_csv_file(edge_data, edge_header)
                    st.download_button("エッジCSVをダウンロード", edge_csv, "edges.csv", "text/csv")

                with col_dl3:
                    is_success, buffer = cv2.imencode(".png", marked_img)
                    if is_success:
                        st.download_button("グラフ画像をダウンロード", buffer.tobytes(), "graph_marked.png", "image/png")

                with st.expander("📊 ノードデータプレビュー"):
                    import pandas as pd
                    df_nodes = pd.DataFrame(node_data, columns=['node_id', 'x_scratch', 'y_scratch', 'type_code', 'type_label'])
                    st.dataframe(df_nodes.head(10))

                with st.expander("🔗 エッジデータプレビュー"):
                    if enable_distance_scale:
                        df_edges = pd.DataFrame(edge_data, columns=['edge_id', 'from_node_id', 'to_node_id', 'pixel_length', 'distance_meters'])
                    else:
                        df_edges = pd.DataFrame(edge_data, columns=['edge_id', 'from_node_id', 'to_node_id', 'pixel_length'])
                    st.dataframe(df_edges.head(10))

else:
    st.info("👆 左のファイルアップローダーから画像を選択してください")
    with st.expander("📖 使い方"):
        st.markdown("""
        ### 使い方
        1. 画像をアップロード  
        2. 距離スケール設定（任意）  
        3. パラメータ調整  
        4. 「グラフデータを生成」をクリック  
        5. 結果確認・ダウンロード  
        """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

