from threading import Thread, Event
import cv2
import torch
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, SlideTransition, Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.resources import resource_add_path, resource_find
from kivy.uix.button import Button
import serial.tools.list_ports


resource_add_path('./data')

# WEIGHT FILES #
weights_path_object_detection = resource_find('best.pt')
weights_path_logo_detection = resource_find('reallogolast.pt')

# MODEL LOADED #
model_object_detection = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path_object_detection)
model_logo_detection = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path_logo_detection)

# DETECTION SETTINGS #
model_object_detection.conf = 0.25
model_object_detection.iou = 0.45
model_object_detection.agnostic = False
model_object_detection.multi_label = False
model_object_detection.max_det = 1

model_logo_detection.conf = 0.25
model_logo_detection.iou = 0.45
model_logo_detection.agnostic = False
model_logo_detection.multi_label = False
model_logo_detection.max_det = 1

# STOP THREADING #
object_detection_thread_stop = Event()
logo_detection_thread_stop = Event()

plastic_detected = False
logo_detected = False
exit_loops = False

logo_class_to_screen = {
    'Pepsi': 'pepsi_screen',
    'sprite': 'sprite_screen',
    'cocacola': 'coca_cola_screen',
    'sevenup': 'seven_up_screen'
}


class MenuScreen(Screen):
    pass

class ManualScreen(Screen):
    pass


class InfoScreen(Screen):
    # Update the descriptions for InfoScreen
    descriptions = [
        "This is an image of Erikaa for InfoScreen.",
        "This is an image of Raiden for InfoScreen.",
        "This is an image of Sprite logo for InfoScreen."
    ]

    def update_description_text(self):
        current_index = self.ids.image_carousel.index
        self.ids.description_text.text = self.descriptions[current_index]

class MachineScreen(Screen):
    # Update the descriptions for MachineScreen
    descriptions = [
        "This is an image of Erikaa for MachineScreen.",
        "This is an image of Raiden for MachineScreen.",
        "This is an image of Sprite logo for MachineScreen."
    ]

    def update_description_text(self):
        current_index = self.ids.image_carousel.index
        self.ids.description_text.text = self.descriptions[current_index]

class GuidinoScreen(Screen):
    # Update the descriptions for GuidinoScreen
    descriptions = [
        "This is an image of Erikaa for GuidinoScreen.",
        "This is an image of Raiden for GuidinoScreen.",
        "This is an image of Sprite logo for GuidinoScreen."
    ]

    def update_description_text(self):
        current_index = self.ids.image_carousel.index
        self.ids.description_text.text = self.descriptions[current_index]
class LogoScreen(Screen):
    pass

class PepsiScreen(Screen):
    pass


class SpriteScreen(Screen):
    pass


class CocaColaScreen(Screen):
    pass


class SevenUpScreen(Screen):
    pass



class IntroScreen(Screen):
    Builder.load_file("intro_scree.kv")

    def on_enter(self):
        print("Entering IntroScreen")
        print(f"plastic_detected: {plastic_detected}")

        if plastic_detected:
            self.ids.start_button.opacity = 1  # Show the button

    def on_leave(self):
        print("Leaving IntroScreen")

    def on_button_press(self):
        self.manager.current = 'menu_screen'


class LogoDetectionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.confirm_popup = None
        self.logo_detected = False
        self.exit_loops = False
        self.cap = cv2.VideoCapture(0)

    def on_enter(self):
        print("Entering LogoDetectionScreen")
        self.ids.status_label.text = "Press 'Start' to run YOLOv5 logo detection..."
        print(f"Using weight file: {weights_path_logo_detection}")
        self.show_confirmation_popup()

    def on_leave(self):
        print("Leaving LogoDetectionScreen")
        self.stop_detection()

    def show_confirmation_popup(self):
        content = BoxLayout(orientation="vertical")
        content.add_widget(Label(text="Do you want to start logo detection?"))
        yes_button = Button(text="Yes")
        yes_button.bind(on_release=self.start_detection)
        no_button = Button(text="No")
        no_button.bind(on_release=self.go_to_menu)
        content.add_widget(yes_button)
        content.add_widget(no_button)
        self.confirm_popup = Popup(title="Confirmation", content=content, size_hint=(0.6, 0.3))
        self.confirm_popup.open()

    def start_detection(self, instance):
        self.exit_loops = False
        self.logo_detected = False
        self.confirm_popup.dismiss()
        self.detect_logos()

    def go_to_menu(self, instance):
        self.manager.current = 'menu_screen'
        self.confirm_popup.dismiss()
        self.stop_detection()

    def stop_detection(self):
        self.exit_loops = True
        if self.cap.isOpened():
            self.cap.release()

    def go_to_detected_screen(self, class_name):

        if class_name in logo_class_to_screen:
            detected_screen = logo_class_to_screen[class_name]
            self.logo_detected = True
            self.manager.current = detected_screen
            self.stop_detection()

    def detect_logos(self):
        detected_classes = []

        while not self.exit_loops:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = model_logo_detection(frame)
            predictions = results.pred[0]

            detected_classes.clear()

            for box, score, category in zip(predictions[:, :4], predictions[:, 4], predictions[:, 5]):
                x1, y1, x2, y2 = box
                class_index = int(category)
                class_name = model_logo_detection.names[class_index]

                if class_name in logo_class_to_screen:
                    print(f"Detected: {class_name}, Score: {score}")
                    self.go_to_detected_screen(class_name)
                    detected_classes.append(class_name)

            cv2.imshow('Logo Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.cap.release()

        if detected_classes:
            print("Logo detection process completed. Detected classes:", detected_classes)
        else:
            print("Logo detection process completed without detecting any logos.")

    def detect_logos_periodically(self, dt):
        if self.logo_detected:
            self.ids.status_label.text = "Logo detected!"
            self.manager.current = 'menu_screen'
            self.stop_detection()

    def update_label(self, dt):
        if self.logo_detected:
            self.ids.status_label.text = "Logo detected!"
            self.manager.current = 'menu_screen'



gizduino_vid_pid = "VID:PID"
def detect_gizduino_ports():
    gizduino_ports = [
        p.device
        for p in serial.tools.list_ports.comports()
        if gizduino_vid_pid in p.hwid
    ]
    return gizduino_ports


class ArduinoCOM(Screen):
    def check_serial_connection(self):
        detected_ports = detect_gizduino_ports()
        result_label = self.ids.result_label  # Access the result_label using ids
        next_screen_button = self.ids.next_screen_button  # Access the next_screen_button using ids

        if detected_ports:
            message = "Arduino detected on ports:\n" + "\n".join(detected_ports)
            result_label.text = message
            next_screen_button.disabled = False
        else:
            message = "No Arduino detected. Please make sure you are connected"
            result_label.text = message

    def switch_to_next_screen(self):
        self.manager.current = "logo_detection_screen"



class PETapp(MDApp):

    def build(self):
        Builder.load_file("menu.kv")
        Builder.load_file("logo_detection_screen.kv")
        Builder.load_file("arduino.kv")
        Builder.load_file("manual_screen.kv")

        self.title = "PET Machine"
        sm = ScreenManager(transition=SlideTransition())

        intro_screen = IntroScreen(name='intro_screen')
        arduino = ArduinoCOM(name='arduino_screen')
        logo_detection_screen = LogoDetectionScreen(name='logo_detection_screen')
        pet_manual_screen = ManualScreen(name='manual_screen')
        logo_sub_menu = LogoScreen(name='logo_sub_menu')
        pepsi_screen = PepsiScreen(name='pepsi_screen')
        sprite_screen = SpriteScreen(name='sprite_screen')
        coca_cola_screen = CocaColaScreen(name='coke_screen')
        seven_up_screen = SevenUpScreen(name='seven_up')
        menu_screen = MenuScreen(name='menu_screen')
        info_screen = InfoScreen()
        machine_screen = MachineScreen()
        guidino_screen = GuidinoScreen()

        sm.add_widget(intro_screen)
        sm.add_widget(arduino)
        sm.add_widget(logo_detection_screen)
        sm.add_widget(pet_manual_screen)
        sm.add_widget(logo_sub_menu)
        sm.add_widget(pepsi_screen)
        sm.add_widget(sprite_screen)
        sm.add_widget(coca_cola_screen)
        sm.add_widget(seven_up_screen)
        sm.add_widget(menu_screen)
        sm.add_widget(info_screen)
        sm.add_widget(machine_screen)
        sm.add_widget(guidino_screen)

        self.logo_detection_schedule = None

        object_detection_thread_stop.clear()
        logo_detection_thread_stop.clear()

        self.object_detection_thread = Thread(target=self.detect_objects)
        self.object_detection_thread.start()

        return sm

    def detect_objects(self):
        global plastic_detected, object_detection_thread_stop
        cap = cv2.VideoCapture(0)

        while not object_detection_thread_stop.is_set():
            ret, frame = cap.read()

            if not ret:
                break

            results = model_object_detection(frame)

            predictions = results.pred[0]
            for box, score, category in zip(predictions[:, :4], predictions[:, 4], predictions[:, 5]):
                x1, y1, x2, y2 = box
                class_index = int(category)
                class_name = model_object_detection.names[class_index]

                if class_name == "plastic":
                    print("Plastic detected!")
                    plastic_detected = True
                    object_detection_thread_stop.set()

                    Clock.schedule_once(self.show_start_button, 0)

                    break

            cv2.imshow('YOLOv5 Live Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        return plastic_detected

    def show_start_button(self, dt):
        self.root.get_screen('intro_screen').ids.start_button.opacity = 1

    def on_stop(self):
        global exit_loops
        exit_loops = True

        if self.object_detection_thread and self.object_detection_thread.is_alive():
            object_detection_thread_stop.set()
            self.object_detection_thread.join()


if __name__ == "__main__":
    PETapp().run()
