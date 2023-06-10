from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
import numpy as np
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg


class FrontPage(Screen):
    def on_enter(self):
        self.box = BoxLayout(orientation='vertical', size_hint=(1, 1))

        # Start button
        start_button = Button(text='Start', size_hint=(1, 0.25), font_size=50)
        start_button.bind(on_release=self.show_second_screen)
        self.box.add_widget(start_button)

        self.add_widget(self.box)

    def show_second_screen(self, instance):
        self.manager.current = 'second_page'


class SecondPage(Screen):
    def on_enter(self):
        self.box = BoxLayout(orientation='vertical', size_hint=(1, 1))

        # Instructions label
        instructions = Label(text='Please input the parameters!', size_hint=(1, 0.15))
        self.box.add_widget(instructions)

        # Container for input boxes
        input_box = BoxLayout(orientation='vertical', size_hint=(1, .3))

        # Input box for N
        self.N_input = TextInput(hint_text='N', size_hint=(1, None), height=90, font_size=50, halign="center",
                                 foreground_color=(1, 0, 0, 1))
        input_box.add_widget(self.N_input)

        # Another BoxLayout for the mean and std_dev inputs
        input_box_row2 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=90)

        # Input box for mean
        self.mean_input = TextInput(hint_text='Mean', size_hint=(0.5, None), height=90, font_size=50,
                                    halign="center", foreground_color=(1, 0, 0, 1))
        self.mean_input.bind(text=self.on_mean_input)
        input_box_row2.add_widget(self.mean_input)

        # Input box for std_dev
        self.std_dev_input = TextInput(hint_text='Standard Deviation', size_hint=(0.5, None), height=90,
                                       disabled=True, font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
        input_box_row2.add_widget(self.std_dev_input)

        input_box.add_widget(input_box_row2)
        self.box.add_widget(input_box)

        # Create Plot button
        button = Button(text='Create Plot', size_hint=(1, 0.25), font_size=50)
        button.bind(on_release=self.create_plot)
        self.box.add_widget(button)

        self.add_widget(self.box)

    def on_mean_input(self, instance, value):
        # Enable/disable the std_dev input based on the mean input
        if value:
            self.std_dev_input.disabled = False
        else:
            self.std_dev_input.disabled = False
            self.std_dev_input.text = ''  # Clear the std_dev input

    def create_plot(self, instance):
        try:
            nn = int(self.N_input.text) if self.N_input.text else None
        except:
            print("Please make sure N is an integer value.")
            return
        try:
            mean = float(self.mean_input.text) if self.mean_input.text else None
        except:
            print("Please make sure the mean is a float value.")
            return
        try:
            std_dev = float(self.std_dev_input.text) if self.std_dev_input.text else None
        except:
            print("Please make sure the standard deviation is a float value.")
            return

        if nn is None or mean is None or std_dev is None:
            print('Please enter all parameters.')
            return

        data = np.random.normal(mean, std_dev, nn)
        plt.hist(data, bins=30)
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        fig = plt.gcf()
        canvas = FigureCanvasKivyAgg(fig)
        self.box.add_widget(canvas)


class MyApp(App):
    def build(self):
        self.screen_manager = ScreenManager()

        front_page = FrontPage(name='front_page')
        self.screen_manager.add_widget(front_page)

        second_page = SecondPage(name='second_page')
        self.screen_manager.add_widget(second_page)

        return self.screen_manager


if __name__ == '__main__':
    MyApp().run()
