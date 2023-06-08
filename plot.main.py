from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
import numpy as np
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt


class MyApp(App):
    def build(self):
        self.box = BoxLayout(orientation='vertical')

        # Instructions label
        instructions = Label(text='Please input the parameters for creating the figure!', size_hint=(1, 0.1))
        self.box.add_widget(instructions)

        # Container for input boxes
        input_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))

        # Input box for mean
        mean_input = TextInput(hint_text='Mean', size_hint=(0.5, None), height=40)
        input_box.add_widget(mean_input)

        # Input box for std_dev
        std_dev_input = TextInput(hint_text='Standard Deviation', size_hint=(0.5, None), height=40)
        input_box.add_widget(std_dev_input)

        self.box.add_widget(input_box)

        # Create Plot button
        button = Button(text='Create Plot', size_hint=(1, 0.2))
        button.bind(on_release=lambda btn: self.create_plot(mean_input.text, std_dev_input.text))
        self.box.add_widget(button)

        return self.box

    def create_plot(self, mean, std_dev):
        try:
            mean = float(mean)
            std_dev = float(std_dev)
            data = np.random.normal(mean, std_dev, size=100)

            fig, ax = plt.subplots()
            ax.plot(data)
            ax.set_ylabel('some numbers')

            mean_value = np.mean(data)  # Calculate the mean of the data

            # Add the dashed line at the mean
            ax.axhline(mean_value, color='red', linestyle='--')

            self.box.add_widget(FigureCanvasKivyAgg(figure=fig))

            plt.show()

        except ValueError:
            print('Invalid input. Please enter numeric values for mean and standard deviation.')


if __name__ == '__main__':
    MyApp().run()
