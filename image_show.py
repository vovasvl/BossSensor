import cv2


def show_image(image_path='s_pycharm.jpg'):
    print('START')
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    # Отображаем изображение в полноэкранном режиме
    cv2.namedWindow("Boss Detected", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Boss Detected", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Boss Detected", image)

    # Ожидание нажатия клавиши (любой клавиши)
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # Escape
            break

    # Закрываем окно с изображением
    cv2.destroyWindow("Boss Detected")
    print('FLAG 4')