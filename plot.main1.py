from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
import numpy as np
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib as mpl


class MyApp(App):
    def build(self):
        self.box = BoxLayout(orientation='vertical', size_hint=(1, 1))

        # Instructions label
        instructions = Label(text='Please input the parameters!', size_hint=(1, 0.15))
        self.box.add_widget(instructions)

        # Container for input boxes
        input_box = BoxLayout(orientation='vertical', size_hint=(1, .3))

        # Input box for N
        self.N_input = TextInput(hint_text='N', size_hint=(1, None), height=90, font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
        input_box.add_widget(self.N_input)

        # Another BoxLayout for the mean and std_dev inputs
        input_box_row2 = BoxLayout(orientation='horizontal', size_hint=(1, None), height=90)

        # Input box for mean
        self.mean_input = TextInput(hint_text='Mean', size_hint=(0.5, None), height=90, font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
        self.mean_input.bind(text=self.on_mean_input)
        input_box_row2.add_widget(self.mean_input)

        # Input box for std_dev
        self.std_dev_input = TextInput(hint_text='Standard Deviation', size_hint=(0.5, None), height=90, disabled=True, font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
        input_box_row2.add_widget(self.std_dev_input)

        input_box.add_widget(input_box_row2)
        self.box.add_widget(input_box)

        # Create Plot button
        button = Button(text='Create Plot', size_hint=(1, 0.25), font_size=50)
        button.bind(on_release=self.create_plot)
        self.box.add_widget(button)

        return self.box


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
            print("Please make sure the N is a float value.")
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

        if mean is None:
            print('Please enter the mean.')
            return

        if std_dev is None:
            print('Please enter the standard deviation.')
            return
        elif std_dev < 0:
            print('Please make sure the standard deviation is not negative.')
            return

        # Clear the existing plot
        for child in self.box.children[:]:
            if isinstance(child, FigureCanvasKivyAgg):
                self.box.remove_widget(child)

        data = np.random.normal(mean, std_dev, size=nn)

        fig, axs = plt.subplots(2, 1)
        mpl.rcParams['font.size'] = 15
        label_font_size = 15

        # Plot the histogram
        axs[0].hist(data, bins='auto', alpha=0.7, rwidth=0.85)
        axs[0].set_ylabel('Frequency', fontsize=label_font_size)

        # Plot the line plot
        axs[1].plot(data)
        axs[1].set_ylabel('Values')

        mean_value = np.mean(data)  # Calculate the mean of the data

        # Add the dashed line at the mean
        axs[1].axhline(mean_value, color='r', linestyle='--', label='Mean')
        axs[1].legend(fontsize=label_font_size)
        axs[1].set_ylabel('Value', fontsize=label_font_size)

        # Create a FigureCanvasKivyAgg and add it to the box layout
        canvas = FigureCanvasKivyAgg(fig)
        self.box.add_widget(canvas)

        plt.close(fig)  # Close the figure to release resources

        # Bind a touch event to the canvas widget to go back to the starting screen
        canvas.bind(on_touch_down=self.go_to_start_screen)

    def go_to_start_screen(self, instance, touch):
        if touch.is_double_tap:
            self.box.clear_widgets()  # Clear all widgets in the box layout
            self.build()  # Rebuild the starting screen

            return True  # Consume the touch event


if __name__ == '__main__':
    MyApp().run()
