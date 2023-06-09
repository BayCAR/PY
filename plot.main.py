from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
import numpy as np
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg


class MyApp(App):
    def build(self):
        self.box = BoxLayout(orientation='vertical')

        # Instructions label
        instructions = Label(text='Please input the parameters for creating the figure!', size_hint=(1, 0.1))
        self.box.add_widget(instructions)

        # Container for input boxes
        input_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))

        # Input box for mean
        self.mean_input = TextInput(hint_text='Mean', size_hint=(0.5, None), height=90, font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
        self.mean_input.bind(text=self.on_mean_input)
        input_box.add_widget(self.mean_input)

        # Input box for std_dev
        self.std_dev_input = TextInput(hint_text='Standard Deviation', size_hint=(0.5, None), height=90, disabled=True, font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
        input_box.add_widget(self.std_dev_input)

        self.box.add_widget(input_box)

        # Create Plot button
        button = Button(text='Create Plot', size_hint=(1, 0.2), font_size=50)
        button.bind(on_release=self.create_plot)
        self.box.add_widget(button)

        return self.box

    def on_mean_input(self, instance, value):
        # Enable/disable the std_dev input based on the mean input
        if value:
            self.std_dev_input.disabled = False
        else:
            self.std_dev_input.disabled = True
            self.std_dev_input.text = ''  # Clear the std_dev input

    def create_plot(self, instance):
        mean = float(self.mean_input.text) if self.mean_input.text else None
        std_dev = float(self.std_dev_input.text) if self.std_dev_input.text else None

        if mean is None:
            print('Please enter the mean.')
            return

        if std_dev is None:
            print('Please enter the standard deviation.')
            return

        data = np.random.normal(mean, std_dev, size=100)

        fig, axs = plt.subplots(2, 1)

        # Plot the histogram
        axs[0].hist(data, bins='auto', alpha=0.7, rwidth=0.85)
        axs[0].set_ylabel('Frequency')

        # Plot the line plot
        axs[1].plot(data)
        axs[1].set_ylabel('Values')

        mean_value = np.mean(data)  # Calculate the mean of the data

        # Add the dashed line at the mean
        axs[1].axhline(mean_value, color='red', linestyle='--')

        # Create a FigureCanvasKivyAgg and add it to the box layout
        canvas = FigureCanvasKivyAgg(fig)
        self.box.add_widget(canvas)

    plt.show()


if __name__ == '__main__':
    MyApp().run()
