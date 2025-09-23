# app_gui_gender.py
# =================
# Giao diện Tkinter “khung chính + sidebar nút + khu vực hiển thị bên phải”.
# - 3 nút: Webcam / Ảnh / Video -> chạy các script infer_* tương ứng (mở cửa sổ OpenCV).
# - Khu vực bên phải hiển thị LOGO mờ (ẩn) – tự co giãn theo cửa sổ.
# - Có vạch ngăn cách giữa cột nút và khu vực hiển thị.
#
# *** Mẹo chỉnh nhanh (xem chi tiết ở từng dòng comment phía dưới):
#   - Độ mờ logo:    LOGO_OPACITY
#   - Độ sáng logo:  LOGO_BRIGHTEN
#   - Kích thước logo:  trong _on_resize() -> hệ số 0.82
#   - Vị trí logo:      trong _on_resize() -> cx, cy (đặc biệt cy = int(h*0.55))
#   - Bật viền tròn:    SHOW_CIRCLE = True
#   - Bề rộng sidebar:  SIDEBAR_W
#   - Độ dày vạch:      SEPARATOR_W
#   - Khoảng cách/đệm:  các tham số padx/pady ở grid() của title/sidebar/right

import os, sys, subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageDraw

# ========= CẤU HÌNH DỄ CHỈNH =========
LOGO_PATH = "Logo_EAUT.png"  # ← Đường dẫn file logo PNG/JPG
LOGO_OPACITY = 0.10          # ← Độ mờ logo (0.0–1.0). 0.08 mờ hơn, 0.15 đậm hơn
LOGO_BRIGHTEN = 1.35         # ← Độ sáng logo (>1 sáng hơn, <1 tối hơn). 1.2–1.5 là đẹp
TITLE_TEXT = "HỆ THỐNG DỰ ĐOÁN GIỚI TÍNH"

WIN_W, WIN_H = 1200, 720     # ← Kích thước cửa sổ ban đầu
SIDEBAR_W = 220              # ← Bề rộng cột nút bên trái
SEPARATOR_W = 2              # ← Độ dày vạch ngăn cách

SHOW_CIRCLE = False          # ← True: vẽ viền tròn ngoài logo; False: chỉ logo

# Hàm chạy 1 script python con (mở process riêng – cửa sổ OpenCV riêng)
def run_py(path, *args):
    try:
        subprocess.run([sys.executable, path, *args], check=False)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không chạy được: {path}\n{e}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # ---- Cửa sổ chính ----
        self.title("Du doan gioi tinh")
        self.geometry(f"{WIN_W}x{WIN_H}")   # ← kích thước ban đầu
        self.minsize(1000, 640)             # ← không cho nhỏ quá gây vỡ layout
        self.configure(bg="#f7f9fc")        # ← màu nền tổng thể

        # ---- Tiêu đề/band đầu trang ----
        title = tk.Label(
            self,
            text=TITLE_TEXT,
            bg="#e9f1ff",                    # ← màu nền band tiêu đề
            fg="#0f172a",                    # ← màu chữ
            font=("Segoe UI", 26, "bold"),   # ← font + cỡ chữ tiêu đề
            pady=10                          # ← đệm dọc trong band
        )
        # padx/pady của grid bên ngoài -> khoảng cách band với viền trên & 2 bên
        title.grid(row=0, column=0, columnspan=3, sticky="ew", padx=20, pady=(15, 10))

        # ---- Sidebar chứa các nút ----
        self.sidebar = tk.Frame(self, bg="#f7f9fc", width=SIDEBAR_W)  # ← bề rộng cột
        # padx/pady ở đây là khoảng cách cột nút với các mép ngoài
        self.sidebar.grid(row=1, column=0, sticky="nsw", padx=(20, 10), pady=(0, 15))
        self.sidebar.grid_propagate(False)  # giữ nguyên width do ta set ở trên

        def mkbtn(text, cmd, accent=False):
            """
            Tạo 1 nút với style đồng bộ.
            - accent=True: dùng màu khác cho nút 'Thoát'
            - Muốn đổi font/cỡ chữ -> đổi dòng font=...
            - Muốn đổi màu nền/hover -> bg, activebackground
            """
            bg = "#e8f1ff" if not accent else "#d6eef6"
            return tk.Button(
                self.sidebar, text=text, command=cmd,
                font=("Segoe UI", 12, "bold"),   # ← cỡ chữ nút
                bg=bg, fg="#0f172a",
                activebackground="#dbeafe",
                relief="groove", bd=2,
                padx=10, pady=10, cursor="hand2"
            )

        # Đặt 3 nút chức năng + 1 nút thoát
        # - Chỉnh khoảng cách dọc giữa các nút bằng pady.
        mkbtn("Dự đoán bằng\nWebcam", self.on_webcam).grid(row=0, column=0, sticky="ew", pady=(0, 18))
        mkbtn("Dự đoán bằng\nảnh",    self.on_image ).grid(row=1, column=0, sticky="ew", pady=18)
        mkbtn("Dự đoán bằng\nVideo",  self.on_video ).grid(row=2, column=0, sticky="ew", pady=18)
        mkbtn("Thoát", self.destroy, accent=True).grid(row=99, column=0, sticky="ew", pady=(30, 0))

        # ---- Vạch ngăn cách ----
        self.sep = tk.Frame(self, bg="#d8dee9", width=SEPARATOR_W)  # ← màu & dày vạch
        self.sep.grid(row=1, column=1, sticky="ns", pady=(0, 15))

        # ---- Khu vực hiển thị bên phải (logo / preview) ----
        self.right = tk.Frame(self, bg="#ffffff")  # màu nền khu hiển thị
        # padding trái/phải của khu vực hiển thị:
        self.right.grid(row=1, column=2, sticky="nsew", padx=(10, 20), pady=(0, 15))

        # Cho cột 2 (right) co giãn – chiếm phần còn lại
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Canvas vẽ logo (và có thể dùng để embed preview sau này)
        self.canvas = tk.Canvas(self.right, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Giữ reference ảnh Tk để tránh bị GC
        self.logo_imgtk = None

        # Mỗi khi resize cửa sổ -> vẽ lại logo cho vừa khung
        self.bind("<Configure>", self._on_resize)

    # ==== Hành động các nút ====
    def on_webcam(self):
        run_py("infer_gender_realtime.py")  # mở process script webcam (cửa sổ OpenCV riêng)

    def on_image(self):
        p = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("Tất cả", "*.*")]
        )
        if p:
            run_py("infer_gender_image.py", p)  # truyền đường dẫn ảnh cho script

    def on_video(self):
        p = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("Tất cả", "*.*")]
        )
        if p:
            run_py("infer_gender_video.py", p)

    # ==== Căn tỉ lệ & vẽ logo mỗi khi thay đổi kích thước ====
    def _on_resize(self, _=None):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

        # ---- KÍCH THƯỚC & VỊ TRÍ LOGO ----
        # * Logo sẽ nằm trong 1 hình tròn đường kính = 0.82 * min(w, h)
        #   -> muốn logo to/nhỏ hơn: đổi 0.82 (vd 0.75 nhỏ, 0.90 to)
        diameter = int(min(w, h) * 0.82)

        # * Tâm logo: cx, cy
        #   - cx = w//2 -> giữa theo chiều ngang
        #   - cy = int(h*0.55) -> cao 55% chiều cao (trên 50% 1 chút để nhìn cân)
        #   -> muốn logo lên/xuống: đổi 0.55 (0.50 ở chính giữa, 0.60 thấp hơn)
        cx, cy = w // 2, int(h * 0.55)

        r = diameter // 2
        x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r

        # Vẽ viền tròn nếu bật SHOW_CIRCLE
        if SHOW_CIRCLE:
            self.canvas.create_oval(x0, y0, x1, y1, outline="#222222", width=2)

        # Vẽ logo (nếu có file)
        if os.path.exists(LOGO_PATH):
            try:
                # trừ 2px nếu có viền để logo không đè viền
                self._draw_logo_circle(LOGO_PATH, cx, cy, r - (2 if SHOW_CIRCLE else 0))
            except Exception as e:
                self.canvas.create_text(cx, cy, text=f"Lỗi logo: {e}",
                                        font=("Segoe UI", 12), fill="red")
        else:
            self.canvas.create_text(cx, cy, text="(Thiếu logo)",
                                    font=("Segoe UI", 14), fill="#6b7280")

    # ==== Scale logo vừa hình tròn + áp alpha mờ ====
    def _draw_logo_circle(self, path, cx, cy, radius):
        img = Image.open(path).convert("RGBA")

        # Điều chỉnh độ sáng (LOGO_BRIGHTEN >1 sáng hơn, <1 tối hơn)
        if LOGO_BRIGHTEN != 1.0:
            img = ImageEnhance.Brightness(img).enhance(LOGO_BRIGHTEN)

        # Fit logo vào hình vuông kích thước (2*radius) mà không méo
        size = radius * 2
        img = ImageOps.contain(img, (size, size), method=Image.LANCZOS)

        # Tạo mask tròn có alpha = LOGO_OPACITY -> logo dạng “mờ mờ ẩn”
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0] - 1, img.size[1] - 1), fill=int(255 * LOGO_OPACITY))
        img.putalpha(mask)

        # Đưa lên canvas
        self.logo_imgtk = ImageTk.PhotoImage(img)
        self.canvas.create_image(cx, cy, image=self.logo_imgtk)

# ---- Điểm vào chương trình ----
if __name__ == "__main__":
    App().mainloop()
