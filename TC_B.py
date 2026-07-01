import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.lines import Line2D
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tcmarkers  

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PAST_CSV = os.path.join(BASE_PATH, "past_track_B.csv")
FORE_CSV = os.path.join(BASE_PATH, "forecast_track_B.csv")
OUTPUT_IMG = os.path.join(BASE_PATH, "TC_forecast_B.png")

# 確保中文字體顯示正常
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_info(wind):
    if wind < 41: return "LPA 低壓區", "#E0E0E0"
    elif 41 <= wind <= 62: return "TD 熱帶低氣壓", "#F9F1A5"
    elif 63 <= wind <= 87: return "TS 熱帶風暴", "#3498DB"
    elif 88 <= wind <= 117: return "STS 強烈熱帶風暴", "#2ECC71"
    elif 118 <= wind <= 149: return "TY 颱風", "#F1C40F"
    elif 150 <= wind <= 184: return "STY 強颱風", "#E67E22"
    else: return "SUTY 超強颱風", "#9B59B6"

def draw_chart():
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成預報...")
    try:
        df_past = pd.read_csv(PAST_CSV)
        df_fore = pd.read_csv(FORE_CSV)
        
        past_data = df_past[['datetime', 'lng', 'lat', 'wind', 'minimum central pressure']].values.tolist()
        curr = past_data[-1]
        
        forecast_data = []
        for _, row in df_fore.iterrows():
            h_val = int(str(row['f_time']).replace('hr', ''))
            forecast_data.append([row['f_time'], row['lng'], row['lat'], row['wind'], h_val, row['minimum central pressure']])

        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        ax.set_extent([115.0, 175.0, 5.0, 50.0], crs=ccrs.PlateCarree())

        # 地圖風格微調
        ax.add_feature(cfeature.LAND, facecolor="#F7F7F2", edgecolor="#7F8C8D", zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="#EBF5FB", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="#34495E", zorder=2)

        # --- 增加經緯度網格線 ---
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                          linewidth=0.5, color='#DCDCDC', linestyle='--', zorder=2)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(90, 181, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 91, 5))
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}

        # 繪製路徑
        ax.plot([d[1] for d in past_data], [d[2] for d in past_data], color="#27AE60", linewidth=4, zorder=4)

        f_hs = [d[4] for d in forecast_data]
        all_h, all_ln, all_lt = [0] + f_hs, [curr[1]] + [d[1] for d in forecast_data], [curr[2]] + [d[2] for d in forecast_data]
        all_er = [0] + [((h // 24) * 100 + (h % 24) * (100 / 24)) * (1/111) for h in f_hs]
        
        ih = np.linspace(0, max(all_h), 100)
        xi = PchipInterpolator(all_h, all_ln)(ih)
        yi = PchipInterpolator(all_h, all_lt)(ih)
        ri = PchipInterpolator(all_h, all_er)(ih)
        
        thetas = np.linspace(0, 2*np.pi, 360)
        ps = [Polygon(np.dstack((xi[i]+ri[i]*np.cos(thetas), yi[i]+ri[i]*np.sin(thetas)))[0]) for i in range(len(ih))]
        poly = unary_union([MultiPolygon([ps[i], ps[i+1]]).convex_hull for i in range(len(ps)-1)])
        ax.add_geometries([poly], ccrs.PlateCarree(), facecolor="#FFF5D7", alpha=0.4, edgecolor="#FFD180", linewidth=0.8, zorder=3)

        ax.plot(xi, yi, color="#3498DB", linewidth=3, linestyle='--', zorder=4)

        _, curr_col = get_info(curr[3])
        ax.plot(curr[1], curr[2], marker=tcmarkers.HU, ms=13, color=curr_col, mec='k', mew=1.2, zorder=10)

        for d in forecast_data:
            _, ln, lt, wd, h, _ = d
            _, col = get_info(wd)
            if h in {24, 48, 72, 96, 120}:
                ax.plot(ln, lt, marker=tcmarkers.HU, ms=10, color=col, mec='k', mew=1, zorder=10)

        # --- 調整後的圖例 (適當加大) ---
        legend_elements = [
            Line2D([0], [0], color='#27AE60', lw=4, label='過去路徑'),
            Line2D([0], [0], color='#3498DB', lw=3, ls='--', label='預報路徑'),
            Line2D([0], [0], marker='s', color='none', label='70% 誤差帶', 
                   markerfacecolor='#FFF5D7', markeredgecolor='#FFD180', markersize=12, alpha=0.6),
            Line2D([0], [0], color='none', label=''), 
            Line2D([0], [0], color='none', label='強度等級：'),
            Line2D([0], [0], marker='o', color='none', label='LPA 低壓區', markerfacecolor='#E0E0E0', markeredgecolor='k', markersize=11),
            Line2D([0], [0], marker='o', color='none', label='TD 熱帶低氣壓', markerfacecolor='#F9F1A5', markeredgecolor='k', markersize=11),
            Line2D([0], [0], marker='o', color='none', label='TS 熱帶風暴', markerfacecolor='#3498DB', markeredgecolor='k', markersize=11),
            Line2D([0], [0], marker='o', color='none', label='STS 強烈熱帶風暴', markerfacecolor='#2ECC71', markeredgecolor='k', markersize=11),
            Line2D([0], [0], marker='o', color='none', label='TY 颱風', markerfacecolor='#F1C40F', markeredgecolor='k', markersize=11),
            Line2D([0], [0], marker='o', color='none', label='STY 強颱風', markerfacecolor='#E67E22', markeredgecolor='k', markersize=11),
            Line2D([0], [0], marker='o', color='none', label='SUTY 超強颱風', markerfacecolor='#9B59B6', markeredgecolor='k', markersize=11),
        ]

        leg = ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.99, 0.02),
                        fontsize=11, frameon=True, facecolor='white', framealpha=0.9, 
                        edgecolor='#BDC3C7', labelspacing=0.6)
        leg.set_zorder(100)

        # 右上角站名 (適當加大)
        ax.text(0.99, 0.98, "粵港澳天氣站 GHMWS", transform=ax.transAxes, ha='right', va='top', 
                fontsize=18, fontweight='bold', color='#2C3E50', alpha=0.8, zorder=20)
        
        # 左上角信息框 (適當加大字體)
        info_text = f"現時位置：{curr[2]}°N, {curr[1]}°E\n時間：{curr[0]}\n近中心最大風速：{curr[3]} kph\n中心氣壓：{curr[4]} hPa"
        ax.text(0.01, 0.98, info_text, transform=ax.transAxes, va='top', fontsize=12, zorder=20,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='#BDC3C7', boxstyle='round,pad=0.5'))

        ax.set_title("西北太平洋未命名熱帶氣旋路徑預報圖", fontsize=28, fontweight='bold', pad=25)

        plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
        plt.close() 
        print(">>> 預報圖已製作完成。")
    except Exception as e:
        print(f"錯誤: {e}")

class CSVHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        # 1. 忽略資料夾本身的變更事件
        if event.is_directory:
            return
            
        # 2. 捕捉包含 修改、建立、移動 在內的所有檔案儲存行為
        if event.event_type in ('modified', 'created', 'moved'):
            # 處理移動/重新命名事件時取得新路徑，其餘取得源路徑
            check_path = event.dest_path if event.event_type == 'moved' else event.src_path
            
            # 精準比對副檔名
            if check_path.endswith(".csv"):
                print(f"[{time.strftime('%H:%M:%S')}] 偵測到關鍵檔案變更 ({event.event_type}): {os.path.basename(check_path)}")
                
                # 3. 延遲 1.0 秒以確保作業系統完全釋放檔案鎖定，防範 Permission Denied 錯誤
                time.sleep(1.0) 
                draw_chart()

if __name__ == "__main__":
    # 啟動時先自動生成第一張圖
    draw_chart() 
    
    event_handler = CSVHandler()
    observer = Observer()
    observer.schedule(event_handler, BASE_PATH, recursive=False)
    observer.start()
    print(f"[{time.strftime('%H:%M:%S')}] 監控服務已啟動，正在監聽資料夾...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止監控服務...")
        observer.stop()
        
    observer.join()
    print(">>> 服務已安全退出。")