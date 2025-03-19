import cv2
from boss_train import Model
from image_show import show_image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = "haarcascade_frontalface_default.xml"
    model = Model()
    model.load()

    boss_detected = False  # Флаг для отслеживания, обнаружен ли босс

    while True:
        _, frame = cap.read()

        # Преобразование в оттенки серого
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Загрузка каскадного классификатора
        cascade = cv2.CascadeClassifier(cascade_path)

        # Обнаружение лиц
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(facerect) > 0:
            print('face detected')
            for rect in facerect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]

                result = model.predict(image)
                print('RES', result)
                if result == 0:  # boss
                    if not boss_detected:  # Открываем окно только один раз
                        print('Boss is approaching')
                        show_image()  # Вызываем show_image
                        boss_detected = True
                else:
                    print('Not boss')
                    boss_detected = False  # Сбрасываем флаг, если босс больше не в кадре

        # Отображение камеры в окне
        cv2.imshow("Camera", frame)

        # Ожидание нажатия клавиши (10 мс)
        k = cv2.waitKey(10)
        # Завершение программы по нажатию Esc в окне камеры
        if k == 27:
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()