import dearpygui.dearpygui as dpg
import DearPyGui_DragAndDrop as dpg_dnd
import DearPyGui_Animations as dpg_anim
import DearPyGui_Theme as dpg_theme
import fonts
import settings
import json
import threading
import queue
from flask import Flask, request, jsonify
import random
from models.db import init_db, SessionLocal
from models.user_behavior import UserBehavior

from gui.auth_window import show_login_window, login_window_tag  # тег окна авторизации

# Очередь для событий от агента
event_queue = queue.Queue()

# Flask-приложение
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data["features"]
        print(json.dumps(data, indent=4))
        prediction = random.randint(1, 5)
        event_queue.put({
            "timestamp": data.get("timestamp"),
            "user_id": data.get("user_id"),
            "features": list(features.values()),
            "danger_level": prediction
        })
        return jsonify({"risk_level": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def run_server():
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)

# Центрирование окна авторизации
def center_login_window():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()
    dpg.set_item_pos(login_window_tag, [(vp_width - 400) // 2, (vp_height - 250) // 2])

# Обработка очереди событий от агента
def update_from_queue():
    db = SessionLocal()
    while not event_queue.empty():
        event = event_queue.get()
        behavior = UserBehavior(
            user_id=event["user_id"],
            feature_vector=str(event["features"]),
            danger_level=event["danger_level"]
        )
        db.add(behavior)
        db.commit()
        print(f"✅ [{event['timestamp']}] User: {event['user_id']} → Risk: {event['danger_level']}")
    db.close()

# DPG setup
dpg.create_context()
dpg_dnd.initialize()
dpg.bind_theme(dpg_theme.initialize())
dpg.bind_font(fonts.load(show=False))
settings.load_settings()

# Запускаем сервер Flask в фоновом потоке
threading.Thread(target=run_server, daemon=True).start()

# Главная функция — вызывается после старта
def main():
    init_db()
    show_login_window()
    center_login_window()

# DearPyGui
dpg.set_frame_callback(1, main)  # вызывает show_login_window()

dpg.setup_dearpygui()
dpg.create_viewport(title="DLP GUI Test",
                    width=fonts.font_size * 30,
                    height=fonts.font_size * 25,
                    clear_color=dpg_theme.get_current_theme_color_value(dpg.mvThemeCol_WindowBg))
dpg.show_viewport()

# 🔁 Основной цикл
while dpg.is_dearpygui_running():
    update_from_queue()
    dpg_anim.update()
    dpg.render_dearpygui_frame()

dpg.destroy_context()