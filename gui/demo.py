import dearpygui.dearpygui as dpg
import DearPyGui_Theme as dpg_theme
from DearPyGui_Addons import CheckBoxSlider

class MainWindow:
    window = None

    def __init__(self):
        with dpg.window(label="Главное окно") as self.window:
            dpg.add_text("Demo: work-in-progress")
            CheckBoxSlider().create()
            dpg_theme.add_theme_picker()
