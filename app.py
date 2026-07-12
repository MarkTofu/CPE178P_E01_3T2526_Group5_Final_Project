"""
Philippine Bill Detector — Flet Desktop App
Targets Flet 0.85.x

Run:
  py app.py

Requires model/saved_model/ from training.
"""

import threading
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
import flet as ft
from PIL import Image

CLASSES   = ['₱20', '₱50', '₱100', '₱500', '₱1000']
COLORS    = {'₱20':'#C96A12','₱50':'#B52C2C','₱100':'#7B3FA0','₱500':'#A07C10','₱1000':'#1B4FD8'}
THRESHOLD = 0.70
IMG_SIZE  = 224

# Load model once at startup
_model     = tf.saved_model.load('model/saved_model')
_infer     = _model.signatures['serving_default']
_input_key = list(_infer.structured_input_signature[1].keys())[0]
_out_key   = list(_infer.structured_outputs.keys())[0]

def predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)[np.newaxis]
    res = _infer(**{_input_key: tf.constant(arr)})
    return res[_out_key].numpy()[0]

# ── UI ────────────────────────────────────────────────────────────────────────

def main(page: ft.Page):
    page.title                = 'Philippine Bill Detector'
    page.bgcolor              = '#F7F6F2'
    page.padding              = 40
    page.scroll               = 'auto'
    page.horizontal_alignment = 'center'
    page.window.width         = 520
    page.window.height        = 760
    page.window.resizable     = False

    # ── Widgets ───────────────────────────────────────────────────

    preview = ft.Image(
        src='', visible=False,
        width=440, height=260,
        fit='contain', border_radius=12,
    )

    denom_text = ft.Text('', size=64, weight='bold',   text_align='center')
    conf_text  = ft.Text('', size=14, color='#6B6A65', text_align='center')

    result_card = ft.Container(
        visible=False,
        bgcolor='#FFFFFF',
        border_radius=14,
        border=ft.Border(
            top=ft.BorderSide(1, '#E5E4DE'),
            bottom=ft.BorderSide(1, '#E5E4DE'),
            left=ft.BorderSide(1, '#E5E4DE'),
            right=ft.BorderSide(1, '#E5E4DE'),
        ),
        width=440,
        padding=28,
        content=ft.Column([
            ft.Text('DETECTED BILL', size=11, weight='w600', color='#A09F98'),
            denom_text,
            conf_text,
        ], horizontal_alignment='center', spacing=8),
    )

    status = ft.Text('', size=13, color='#A09F98')

    # ── Logic ─────────────────────────────────────────────────────

    def handle_file(path):
        preview.src         = path
        preview.visible     = True
        result_card.visible = False
        status.value        = 'Analyzing…'
        page.update()

        probs   = predict(path)
        max_idx = int(np.argmax(probs))
        max_p   = float(probs[max_idx])

        if max_p < THRESHOLD:
            denom_text.value = 'Not a bill'
            denom_text.color = '#1A1A18'
            denom_text.size  = 36
            conf_text.value  = f'Best match: {max_p*100:.1f}% — below detection threshold'
        else:
            label            = CLASSES[max_idx]
            denom_text.value = label
            denom_text.color = COLORS[label]
            denom_text.size  = 64
            conf_text.value  = f'{max_p*100:.1f}% confident'

        result_card.visible = True
        status.value        = ''
        page.update()

    def on_upload_click(e):
        # Run tkinter dialog in a thread so it doesn't block Flet's event loop
        def pick():
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', True)   # dialog appears on top
            path = filedialog.askopenfilename(
                title='Select a bill photo',
                filetypes=[('Image files', '*.jpg *.jpeg *.png')]
            )
            root.destroy()
            if path:
                handle_file(path)

        threading.Thread(target=pick, daemon=True).start()

    upload_btn = ft.Button('Upload a Photo', on_click=on_upload_click)

    # ── Layout ────────────────────────────────────────────────────

    page.add(
        ft.Column([
            ft.Column([
                ft.Text('Philippine Bill Detector', size=22, weight='w600', color='#1A1A18'),
                ft.Text('Upload a photo of a Philippine paper bill to identify it.',
                        size=14, color='#6B6A65'),
            ], horizontal_alignment='center', spacing=6),

            ft.Container(height=24),
            ft.Row([upload_btn], alignment='center'),
            ft.Container(height=8),
            ft.Row([preview],    alignment='center'),
            result_card,
            ft.Row([status],     alignment='center'),

        ], horizontal_alignment='center', spacing=16)
    )

ft.run(main)
