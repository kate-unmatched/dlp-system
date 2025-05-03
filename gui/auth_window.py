import dearpygui.dearpygui as dpg
from gui import MainWindow  # для загрузки интерфейса после входа

login_window_tag = "login_window"


def login_callback():
    username = dpg.get_value("login_input")
    password = dpg.get_value("password_input")

    # Пример простого словаря, замените позже на БД
    if username == "admin" and password == "admin123":
        dpg.delete_item(login_window_tag)
        main_window = MainWindow()
        dpg.set_primary_window(main_window.window, True)
    else:
        dpg.set_value("login_status", "Неверный логин или пароль")


def show_login_window():
    with dpg.window(tag=login_window_tag, no_title_bar=True, no_resize=True,
                    no_move=True, no_close=True, pos=[0, 0], width=dpg.get_viewport_width(),
                    height=dpg.get_viewport_height()):
        # Центровочный child — внутрь вписываются все элементы
        with dpg.child_window(width=-1, height=-1):
            dpg.add_spacer(height=100)
            dpg.add_text("Авторизация", wrap=600, bullet=False, color=(255, 255, 255), indent=200)
            dpg.add_spacer(height=20)
            dpg.add_input_text(label="Логин", tag="login_input", width=300, indent=150)
            dpg.add_input_text(label="Пароль", tag="password_input", password=True, width=300, indent=150)
            dpg.add_spacer(height=10)
            dpg.add_button(label="Войти", callback=login_callback, width=150, indent=225)
            dpg.add_text("", tag="login_status", color=(255, 0, 0), indent=200)
