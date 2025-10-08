# HeThongNhanDienTuoiGioiTinh.py
import os, sys, subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageDraw

# ========= CẤU HÌNH =========
LOGO_PATH = "Logo_EAUT.png"
LOGO_OPACITY = 0.10
LOGO_BRIGHTEN = 1.35
TITLE_TEXT = "HỆ THỐNG DỰ ĐOÁN TUỔI & GIỚI TÍNH"

WIN_W, WIN_H = 1200, 720
SIDEBAR_W = 220
SEPARATOR_W = 2
SHOW_CIRCLE = False

def run_py(path, *args):
    """Chạy file Python con bằng subprocess"""
    try:
        subprocess.run([sys.executable, path, *args], check=False)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không chạy được: {path}\n{e}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dự đoán Tuổi & Giới tính")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1000, 640)
        self.configure(bg="#f7f9fc")

        # ====== TIÊU ĐỀ ======
        title = tk.Label(
            self, text=TITLE_TEXT, bg="#e9f1ff", fg="#0f172a",
            font=("Segoe UI", 26, "bold"), pady=10
        )
        title.grid(row=0, column=0, columnspan=3, sticky="ew", padx=20, pady=(15, 10))

        # ====== SIDEBAR ======
        self.sidebar = tk.Frame(self, bg="#f7f9fc", width=SIDEBAR_W)
        self.sidebar.grid(row=1, column=0, sticky="nsw", padx=(20, 10), pady=(0, 15))
        self.sidebar.grid_propagate(False)

        def mkbtn(text, cmd, accent=False):
            bg = "#e8f1ff" if not accent else "#d6eef6"
            return tk.Button(
                self.sidebar, text=text, command=cmd,
                font=("Segoe UI", 12, "bold"),
                bg=bg, fg="#0f172a",
                activebackground="#dbeafe",
                relief="groove", bd=2,
                padx=10, pady=10, cursor="hand2"
            )

        mkbtn("Dự đoán bằng\nWebcam", self.on_webcam).grid(row=0, column=0, sticky="ew", pady=(0, 18))
        mkbtn("Dự đoán bằng\nẢnh",    self.on_image ).grid(row=1, column=0, sticky="ew", pady=18)
        mkbtn("Dự đoán bằng\nVideo",  self.on_video ).grid(row=2, column=0, sticky="ew", pady=18)
        mkbtn("Thoát", self.destroy, accent=True).grid(row=99, column=0, sticky="ew", pady=(30, 0))

        # ====== NGĂN CÁCH ======
        self.sep = tk.Frame(self, bg="#d8dee9", width=SEPARATOR_W)
        self.sep.grid(row=1, column=1, sticky="ns", pady=(0, 15))

        # ====== KHUNG BÊN PHẢI ======
        self.right = tk.Frame(self, bg="#ffffff")
        self.right.grid(row=1, column=2, sticky="nsew", padx=(10, 20), pady=(0, 15))
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.canvas = tk.Canvas(self.right, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.logo_imgtk = None
        self.bind("<Configure>", self._on_resize)

    # ====== NÚT CHỨC NĂNG ======
    def on_webcam(self):
        run_py("infer_realtime.py")   # gọi file realtime

    def on_image(self):
        p = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("Tất cả", "*.*")]
        )
        if p:
            run_py("infer_image.py", p)  # gọi file ảnh

    def on_video(self):
        p = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("Tất cả", "*.*")]
        )
        if p:
            run_py("infer_video.py", p)  # gọi file video

    # ====== VẼ LOGO ======
    def _on_resize(self, _=None):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        diameter = int(min(w, h) * 0.82)
        cx, cy = w // 2, int(h * 0.55)
        r = diameter // 2
        x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r

        if SHOW_CIRCLE:
            self.canvas.create_oval(x0, y0, x1, y1, outline="#222222", width=2)

        if os.path.exists(LOGO_PATH):
            try:
                self._draw_logo_circle(LOGO_PATH, cx, cy, r - (2 if SHOW_CIRCLE else 0))
            except Exception as e:
                self.canvas.create_text(cx, cy, text=f"Lỗi logo: {e}",
                                        font=("Segoe UI", 12), fill="red")
        else:
            self.canvas.create_text(cx, cy, text="(Thiếu logo)",
                                    font=("Segoe UI", 14), fill="#6b7280")

    def _draw_logo_circle(self, path, cx, cy, radius):
        img = Image.open(path).convert("RGBA")
        if LOGO_BRIGHTEN != 1.0:
            img = ImageEnhance.Brightness(img).enhance(LOGO_BRIGHTEN)

        size = radius * 2
        img = ImageOps.contain(img, (size, size), method=Image.LANCZOS)

        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0] - 1, img.size[1] - 1),
                     fill=int(255 * LOGO_OPACITY))
        img.putalpha(mask)

        self.logo_imgtk = ImageTk.PhotoImage(img)
        self.canvas.create_image(cx, cy, image=self.logo_imgtk)

if __name__ == "__main__":
    App().mainloop()
