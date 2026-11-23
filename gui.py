"""
All-in-One Modern Desktop UI for AI vs Real Image Detection.
Tabs:
 1. 🔍 Detection Tab: Browse & predict single image with confidence meter and indicators.
 2. 🏋️ Training Tab: Select dataset directory, choose model, set epochs, and start training with live progress.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F

from detect import get_inference_transform
from model import build_model
from dataset import get_dataloaders
import torch.nn as nn
import torch.optim as optim

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI vs Real Image Detector Studio")
        self.geometry("780x750")
        self.minsize(700, 680)
        self.configure(bg="#181825")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = ["FAKE", "REAL"]
        self.model_type = "resnet18"
        self.img_size = 32

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#181825", foreground="#cdd6f4", font=("Segoe UI", 10))
        self.style.configure("TNotebook", background="#181825", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4", padding=[18, 8], font=("Segoe UI", 11, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", "#89b4fa")], foreground=[("selected", "#11111b")])
        self.style.configure("TProgressbar", thickness=16, troughcolor="#313244", background="#a6e3a1")

        self.build_ui()
        self.try_load_model()

    def build_ui(self):
        # Header
        header_frame = tk.Frame(self, bg="#181825")
        header_frame.pack(fill="x", padx=20, pady=(15, 5))

        title = tk.Label(
            header_frame, 
            text="🛡️ AI & Fake Image Detector", 
            font=("Segoe UI", 18, "bold"), 
            fg="#cdd6f4", 
            bg="#181825"
        )
        title.pack(side="left")

        self.device_badge = tk.Label(
            header_frame,
            text=f"Hardware: {self.device.type.upper()}",
            font=("Segoe UI", 9, "bold"),
            bg="#313244",
            fg="#89dceb",
            padx=10,
            pady=4
        )
        self.device_badge.pack(side="right")

        # Tabs container
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=20, pady=15)

        self.detect_tab = tk.Frame(notebook, bg="#1e1e2e")
        self.train_tab = tk.Frame(notebook, bg="#1e1e2e")

        notebook.add(self.detect_tab, text="  🔍 Detect Image  ")
        notebook.add(self.train_tab, text="  🏋️ Train Model  ")

        self.setup_detect_tab()
        self.setup_train_tab()

    # -------------------------------------------------------------
    # TAB 1: DETECTION
    # -------------------------------------------------------------
    def setup_detect_tab(self):
        parent = self.detect_tab

        # Top Model Status Bar
        self.model_status_bar = tk.Label(
            parent,
            text="Checking model status...",
            bg="#313244",
            fg="#fab387",
            font=("Segoe UI", 10),
            padx=12,
            pady=6
        )
        self.model_status_bar.pack(fill="x", padx=20, pady=(15, 10))

        # Main horizontal split
        content_frame = tk.Frame(parent, bg="#1e1e2e")
        content_frame.pack(fill="both", expand=True, padx=20, pady=5)

        # Left: Image display
        left_box = tk.Frame(content_frame, bg="#1e1e2e")
        left_box.pack(side="left", fill="both", expand=True, padx=10)

        self.img_card = tk.Frame(left_box, bg="#313244", width=330, height=330)
        self.img_card.pack_propagate(False)
        self.img_card.pack(pady=10)

        self.img_preview = tk.Label(
            self.img_card, 
            text="No Image Selected\n\nClick 'Choose Image' below", 
            fg="#9399b2", 
            bg="#313244", 
            font=("Segoe UI", 11)
        )
        self.img_preview.pack(expand=True)

        self.pick_btn = tk.Button(
            left_box,
            text="📂 Choose Image to Scan",
            font=("Segoe UI", 11, "bold"),
            bg="#89b4fa",
            fg="#11111b",
            activebackground="#b4befe",
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=8,
            command=self.browse_and_predict
        )
        self.pick_btn.pack(pady=10)

        # Right: Prediction Results Card
        right_box = tk.Frame(content_frame, bg="#313244", padx=20, pady=20)
        right_box.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        res_title = tk.Label(right_box, text="Analysis Results", font=("Segoe UI", 13, "bold"), fg="#cdd6f4", bg="#313244")
        res_title.pack(anchor="w")

        self.verdict_badge = tk.Label(
            right_box,
            text="WAITING FOR IMAGE",
            font=("Segoe UI", 15, "bold"),
            bg="#45475a",
            fg="#bac2de",
            padx=15,
            pady=10
        )
        self.verdict_badge.pack(fill="x", pady=15)

        self.confidence_label = tk.Label(
            right_box,
            text="Confidence: -- %",
            font=("Segoe UI", 12),
            fg="#cdd6f4",
            bg="#313244"
        )
        self.confidence_label.pack(anchor="w", pady=5)

        self.conf_bar = ttk.Progressbar(right_box, orient="horizontal", length=220, mode="determinate")
        self.conf_bar.pack(fill="x", pady=5)

        sep = ttk.Separator(right_box, orient="horizontal")
        sep.pack(fill="x", pady=15)

        self.details_label = tk.Label(
            right_box,
            text="Class Probabilities:\n• Real: --%\n• Fake / AI: --%",
            justify="left",
            font=("Segoe UI", 11),
            fg="#a6adc8",
            bg="#313244"
        )
        self.details_label.pack(anchor="w", pady=5)

    def try_load_model(self):
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            self.model_status_bar.config(
                text="⚠️ No trained model found ('best_model.pth'). Switch to 'Train Model' tab to train one!",
                bg="#45475a",
                fg="#f9e2af"
            )
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classes = checkpoint.get("classes", ["FAKE", "REAL"])
            self.model_type = checkpoint.get("model_type", "resnet18")
            self.img_size = checkpoint.get("img_size", 32)
            val_acc = checkpoint.get("best_acc", 0.0)

            self.model = build_model(model_type=self.model_type, num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()

            acc_str = f" | Val Acc: {val_acc*100:.2f}%" if val_acc else ""
            self.model_status_bar.config(
                text=f"✓ Model Ready: {self.model_type.upper()} ({', '.join(self.classes)}){acc_str}",
                bg="#1e3a2f",
                fg="#a6e3a1"
            )
        except Exception as e:
            self.model_status_bar.config(text=f"Error loading model: {e}", bg="#3b1d28", fg="#f38ba8")

    def browse_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="Select Image to Analyze",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp")]
        )
        if not file_path:
            return

        img = Image.open(file_path).convert("RGB")
        preview = img.copy()
        preview.thumbnail((300, 300), Image.Resampling.NEAREST if img.size[0] <= 64 else Image.Resampling.BICUBIC)
        photo = ImageTk.PhotoImage(preview)
        self.img_preview.config(image=photo, text="")
        self.img_preview.image = photo

        if self.model is None:
            messagebox.showwarning(
                "Model Not Trained", 
                "No trained model ('best_model.pth') loaded.\nPlease train a model in the 'Train Model' tab first."
            )
            return

        transform = get_inference_transform(img_size=self.img_size)
        input_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        label = self.classes[pred_idx.item()].upper()
        confidence = conf.item() * 100

        # Update Result UI
        if "FAKE" in label:
            self.verdict_badge.config(text="⚠️ AI GENERATED / FAKE", bg="#f38ba8", fg="#11111b")
        else:
            self.verdict_badge.config(text="✓ REAL AUTHENTIC IMAGE", bg="#a6e3a1", fg="#11111b")

        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        self.conf_bar["value"] = confidence

        breakdown = ["Class Probabilities:"]
        for idx, cls in enumerate(self.classes):
            breakdown.append(f"• {cls.upper()}: {probs[idx].item() * 100:.2f}%")
        self.details_label.config(text="\n".join(breakdown))

    # -------------------------------------------------------------
    # TAB 2: TRAINING
    # -------------------------------------------------------------
    def setup_train_tab(self):
        parent = self.train_tab

        frame = tk.Frame(parent, bg="#1e1e2e", padx=30, pady=20)
        frame.pack(fill="both", expand=True)

        # Dataset selector
        ds_lbl = tk.Label(frame, text="1. Dataset Folder (CIFAKE or custom):", font=("Segoe UI", 11, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        ds_lbl.grid(row=0, column=0, sticky="w", pady=(0, 5))

        ds_row = tk.Frame(frame, bg="#1e1e2e")
        ds_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        self.dataset_entry = tk.Entry(ds_row, font=("Segoe UI", 10), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4", relief="flat")
        self.dataset_entry.insert(0, os.path.abspath("dataset"))
        self.dataset_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 10))

        ds_browse_btn = tk.Button(
            ds_row, text="Browse...", command=self.browse_dataset,
            bg="#45475a", fg="#cdd6f4", relief="flat", padx=12, pady=4, cursor="hand2"
        )
        ds_browse_btn.pack(side="right")

        # Training Options
        opt_lbl = tk.Label(frame, text="2. Hyperparameters & Architecture:", font=("Segoe UI", 11, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        opt_lbl.grid(row=2, column=0, sticky="w", pady=(0, 5))

        opt_grid = tk.Frame(frame, bg="#313244", padx=15, pady=15)
        opt_grid.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        # Model type
        tk.Label(opt_grid, text="Model Architecture:", bg="#313244", fg="#cdd6f4").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_choice = ttk.Combobox(opt_grid, values=["resnet18", "simple"], state="readonly", width=14)
        self.model_choice.set("resnet18")
        self.model_choice.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Epochs
        tk.Label(opt_grid, text="Epochs:", bg="#313244", fg="#cdd6f4").grid(row=0, column=2, sticky="w", padx=15, pady=5)
        self.epochs_entry = tk.Entry(opt_grid, width=8, bg="#1e1e2e", fg="#cdd6f4", relief="flat")
        self.epochs_entry.insert(0, "5")
        self.epochs_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        # Batch Size
        tk.Label(opt_grid, text="Batch Size:", bg="#313244", fg="#cdd6f4").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.batch_entry = tk.Entry(opt_grid, width=16, bg="#1e1e2e", fg="#cdd6f4", relief="flat")
        self.batch_entry.insert(0, "64")
        self.batch_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Learning Rate
        tk.Label(opt_grid, text="Learning Rate:", bg="#313244", fg="#cdd6f4").grid(row=1, column=2, sticky="w", padx=15, pady=5)
        self.lr_entry = tk.Entry(opt_grid, width=8, bg="#1e1e2e", fg="#cdd6f4", relief="flat")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # Start button
        self.train_btn = tk.Button(
            frame,
            text="🚀 Start Training Model",
            font=("Segoe UI", 12, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            activebackground="#94e2d5",
            relief="flat",
            cursor="hand2",
            pady=10,
            command=self.start_training_thread
        )
        self.train_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        # Progress bar & Log Output
        self.train_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.train_progress.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.log_box = tk.Text(frame, height=12, bg="#11111b", fg="#a6adc8", font=("Consolas", 9), relief="flat")
        self.log_box.grid(row=6, column=0, columnspan=2, sticky="nsew")

        frame.rowconfigure(6, weight=1)
        frame.columnconfigure(0, weight=1)

    def browse_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Root Folder")
        if folder:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, folder)

    def append_log(self, text):
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)

    def start_training_thread(self):
        self.train_btn.config(state="disabled", text="⏳ Training in Progress...", bg="#45475a")
        self.log_box.delete("1.0", tk.END)
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        try:
            data_dir = self.dataset_entry.get().strip()
            epochs = int(self.epochs_entry.get().strip())
            batch_size = int(self.batch_entry.get().strip())
            lr = float(self.lr_entry.get().strip())
            model_type = self.model_choice.get().strip()

            self.append_log(f"Loading dataset from: {data_dir}...")
            train_loader, val_loader, class_names = get_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                img_size=32,
                num_workers=0
            )
            self.append_log(f"Found classes: {class_names}")

            model = build_model(model_type=model_type, num_classes=len(class_names)).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            best_val_acc = 0.0

            for epoch in range(1, epochs + 1):
                self.append_log(f"\n--- Epoch {epoch}/{epochs} ---")
                model.train()
                t_loss, t_corr, total_t = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    t_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    t_corr += torch.sum(preds == labels.data).item()
                    total_t += labels.size(0)

                # Validation
                model.eval()
                v_corr, total_v = 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        v_corr += torch.sum(preds == labels.data).item()
                        total_v += labels.size(0)

                val_acc = v_corr / total_v if total_v > 0 else 0
                train_acc = t_corr / total_t if total_t > 0 else 0
                self.append_log(f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

                pct = (epoch / epochs) * 100
                self.train_progress["value"] = pct

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint = {
                        "model_type": model_type,
                        "model_state": model.state_dict(),
                        "classes": class_names,
                        "best_acc": best_val_acc,
                        "img_size": 32
                    }
                    torch.save(checkpoint, "best_model.pth")
                    self.append_log(f"★ Saved best model checkpoint! ({val_acc*100:.2f}%)")

            self.append_log(f"\n✓ Training Completed! Top Accuracy: {best_val_acc*100:.2f}%")
            self.try_load_model()
            messagebox.showinfo("Training Completed", f"Training finished successfully!\nBest Accuracy: {best_val_acc*100:.2f}%")

        except Exception as e:
            self.append_log(f"\n❌ Error during training: {e}")
            messagebox.showerror("Training Error", str(e))
        finally:
            self.train_btn.config(state="normal", text="🚀 Start Training Model", bg="#a6e3a1")

if __name__ == "__main__":
    app = App()
    app.mainloop()
