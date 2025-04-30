import dearpygui.dearpygui as dpg
import DearPyGui_DragAndDrop as dpg_dnd

import DearPyGui_Animations as dpg_anim
import DearPyGui_Theme as dpg_theme
import fonts
import settings

from models.db import init_db, SessionLocal
from models.user_behavior import UserBehavior

dpg.create_context()
dpg_dnd.initialize()
dpg.bind_theme(dpg_theme.initialize())
dpg.bind_font(fonts.load(show=False))
settings.load_settings()


def main():
    import gui
    main_widow = gui.MainWindow()
    dpg.set_primary_window(main_widow.window, True)
    init_db()
    db = SessionLocal()
    new_behavior = UserBehavior(
        user_id="user42",
        feature_vector="[0.5, 0.8, 0.1]",
        danger_level=4
    )

    db.add(new_behavior)
    db.commit()
    db.close()

    print("✅ Новая запись добавлена в таблицу user_behavior.")


dpg.set_frame_callback(1, main)
dpg.setup_dearpygui()
dpg.create_viewport(title="Demo WIP", width=fonts.font_size * 30, height=fonts.font_size * 25,
                    clear_color=dpg_theme.get_current_theme_color_value(dpg.mvThemeCol_WindowBg))
dpg.show_viewport()

while dpg.is_dearpygui_running():
    dpg_anim.update()
    dpg.render_dearpygui_frame()
dpg.destroy_context()
