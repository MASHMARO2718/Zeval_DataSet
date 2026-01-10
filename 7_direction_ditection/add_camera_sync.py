import sys

# ファイルを読み込み
with open('interactive_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 挿入するコード
new_callback = '''
# カメラドロップダウンからStoreへの同期
@app.callback(
    Output('selected-camera-store', 'data', allow_duplicate=True),
    Input('camera-dropdown', 'value'),
    prevent_initial_call=True
)
def sync_camera_dropdown_to_store(camera):
    """カメラドロップダウンの選択をStoreに同期"""
    return camera

'''

# 挿入位置を探す
insert_marker = '# 折りたたみボタンのコールバック'
pos = content.find(insert_marker)

if pos == -1:
    print("挿入位置が見つかりません")
    sys.exit(1)

# 新しいコードを挿入
new_content = content[:pos] + new_callback + content[pos:]

# ファイルに書き込み
with open('interactive_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✓ カメラドロップダウン同期コールバックを追加しました")




