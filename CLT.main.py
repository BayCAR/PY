from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.core.audio import SoundLoader
from kivy.uix.slider import Slider
from kivy.uix.screenmanager import ScreenManager, Screen
import numpy as np
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib as mpl

current_type = 0


def go_to_previous_screen(self, instance):
    app = App.get_running_app()
    app.root.current = 'home_page'  # Assuming the previous screen's name is 'home_page'


class HomePage(Screen):
    def on_enter(self):
        # self.box = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.box = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.scroll_box = ScrollView()
        self.box.bind(minimum_height=self.box.setter('height'))
        button1 = Button(text='Normal', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button1.background_color = (1, 0, 0, 1)
        button2 = Button(text='Poisson', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button2.background_color = (1, 0.7, 0, 1)
        button3 = Button(text='Binomial', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button3.background_color = (0.7, 1, 0, 1)
        button4 = Button(text='Beta', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button4.background_color = (0, 1, 0, 1)
        button5 = Button(text='Neg. Binomial', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button5.background_color = (0, 1, 0.7, 1)
        button6 = Button(text='Uniform', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button6.background_color = (0, 0.7, 1, 1)
        button7 = Button(text='Chisquare', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button7.background_color = (0, 0, 1, 1)
        button8 = Button(text='Exponential', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button8.background_color = (0.7, 0, 1, 1)
        button9 = Button(text='Standard T', size_hint=(0.95, None), size=(100, 160), pos_hint={'center_x': 0.5},
                         font_size=80)
        button9.background_color = (1, 0, 0.7, 1)

        welcome_text = Label(text='Created by Banana Shaped Cow Studios', size_hint=(1, None), size=(100, 200),
                             pos_hint={'center_x': 0.5},
                             font_size=35, color=(0.2, 0.8, 0.2))
        welcome_description = Label(text='The app offers visualizations of various important probability '
                                         'distributions, making it a convenient and engaging way to utilize a '
                                         'two-minute break.', size_hint=(0.9, None), size=(100, 350),
                                    pos_hint={'center_x': 0.5,
                                              'center_y': 0.0},
                                    font_size=50, color=(0.7, 0.8, .8), halign='center')
        welcome_description.bind(
            width=lambda *x: welcome_description.setter('text_size')(welcome_description, (welcome_description.width,
                                                                                           None)))
        welcome_description.bind(
            height=lambda *x: welcome_description.setter('text_size')(welcome_description, (welcome_description.height,
                                                                                            None)))

        button1.bind(on_release=lambda instance: self.on_button_release(instance, 1))
        button2.bind(on_release=lambda instance: self.on_button_release(instance, 2))
        button3.bind(on_release=lambda instance: self.on_button_release(instance, 3))
        button4.bind(on_release=lambda instance: self.on_button_release(instance, 4))
        button5.bind(on_release=lambda instance: self.on_button_release(instance, 5))
        button6.bind(on_release=lambda instance: self.on_button_release(instance, 6))
        button7.bind(on_release=lambda instance: self.on_button_release(instance, 7))
        button8.bind(on_release=lambda instance: self.on_button_release(instance, 8))
        button9.bind(on_release=lambda instance: self.on_button_release(instance, 9))
        self.box.add_widget(welcome_text)
        self.box.add_widget(welcome_description)
        self.box.add_widget(button1)
        self.box.add_widget(button2)
        self.box.add_widget(button3)
        self.box.add_widget(button4)
        self.box.add_widget(button5)
        self.box.add_widget(button6)
        self.box.add_widget(button7)
        self.box.add_widget(button8)
        self.box.add_widget(button9)
        self.scroll_box.add_widget(self.box)
        self.add_widget(self.scroll_box)

    def on_button_release(self, instance, page_number):
        global current_type
        current_type = page_number
        self.manager.current = 'first_page'

    def on_leave(self):
        # Clear previous content
        self.clear_widgets()

        # Remove all cached screens
        # self.manager.clear_widgets()


class FirstPage(Screen):
    def on_enter(self):
        from kivy.base import EventLoop
        EventLoop.window.bind(on_keyboard=self.hook_keyboard)
        global current_type
        global current_type_name
        current_type_name = 'empty'
        global needed_inputs
        needed_inputs = 0
        global first_input_name
        first_input_name = 'empty'
        global second_input_name
        second_input_name = 'empty'
        global first_input_needs_int
        first_input_needs_int = False
        global second_input_needs_int
        second_input_needs_int = False
        if current_type == 1:
            current_type_name = 'Normal'
            needed_inputs = 2
            first_input_name = 'Mean (μ)'
            second_input_name = 'Standard Division (σ)'
        elif current_type == 2:
            current_type_name = 'Poisson'
            needed_inputs = 1
            first_input_name = 'Rate (λ)'
        elif current_type == 3:
            current_type_name = 'Binomial'
            needed_inputs = 2
            first_input_name = 'Experiment Count (n)'
            first_input_needs_int = True
            second_input_name = 'Success Rate (p)'
        elif current_type == 4:
            current_type_name = 'Beta'
            needed_inputs = 2
            first_input_name = 'Shape 1 (α)'
            first_input_needs_int = True
            second_input_name = 'Shape 2 (β)'
            second_input_needs_int = True
        elif current_type == 5:
            current_type_name = 'Neg. Binomial'
            needed_inputs = 2
            first_input_name = 'Success Count (r)'
            first_input_needs_int = True
            second_input_name = 'Success Rate (p)'
        elif current_type == 6:
            current_type_name = 'Uniform'
            needed_inputs = 2
            first_input_name = 'Lower Bound (a)'
            second_input_name = 'Upper Bound (b)'
        elif current_type == 7:
            current_type_name = 'Chisquare'
            needed_inputs = 1
            first_input_name = 'Degrees (κ)'
            first_input_needs_int = True
        elif current_type == 8:
            current_type_name = 'Exponential'
            needed_inputs = 1
            first_input_name = 'Scale (λ)'
        elif current_type == 9:
            current_type_name = 'Standard t'
            needed_inputs = 1
            first_input_name = 'Degree (ν)'
            first_input_needs_int = True
        self.box = BoxLayout(orientation='vertical', spacing=10)

        # Instructions label
        instructions = Label(text=f'{current_type_name} Distribution:',
                             size_hint=(1, 0.15))

        # possible description label
        description = Button(text='Show Description', size_hint=(0.4, 0.1), pos_hint={'center_x': 0.5})
        description.bind(on_release=self.show_popup)

        # possible error label
        # self.errors = Label(text='Please input Number of simulations and distribution parameters.', size_hint=(1, 0.15), color=(1, 0, 1))
        self.errors = Label(text=' ', size_hint=(1, 0.15), color=(1, 0, 0))

        self.box.add_widget(instructions)
        self.box.add_widget(description)
        self.box.add_widget(self.errors)

        global audio_1
        global audio_2
        audio_1 = SoundLoader.load('CLT_easy.wav')
        audio_2 = SoundLoader.load('CLT_difficult.wav')

        self.audio_1 = audio_1
        self.audio_2 = audio_2

        # Container for input boxes
        input_box = BoxLayout(orientation='vertical', spacing=10, size_hint=(1, .3))
        self.N_input=10000
        title_label = Label(text='Select an Option:', size_hint=(1, 1))
        self.N_input = Spinner(
            # default value shown
            text='Number of Simulations\n(default is 10000). Click to change',
            halign = 'center',
            # available values
            values=('100', '1000', '5000', '10000', '100000'),
            # just for positioning in our example
            size_hint=(0.95, None),
            size=(90, 180),
            font_size=50,
            pos_hint={'center_x': .5, 'center_y': .5})

        # Another BoxLayout for the mean and std_dev inputs
        input_box_row2 = BoxLayout(orientation='horizontal', spacing=10, size_hint=(0.95, None), height=90, pos_hint={
            'center_x': 0.5})

        if needed_inputs == 1:
            self.first_input = TextInput(hint_text=first_input_name, size_hint=(1, None), height=90, font_size=50,
                                         halign="center", foreground_color=(1, 0, 0, 1))
            input_box_row2.add_widget(self.first_input)
        elif needed_inputs == 2:
            self.first_input = TextInput(hint_text=first_input_name, size_hint=(0.5, None), height=90, font_size=50,
                                         halign="center", foreground_color=(1, 0, 0, 1))
            input_box_row2.add_widget(self.first_input)

            self.second_input = TextInput(hint_text=second_input_name, size_hint=(0.5, None), height=90,
                                          font_size=50, halign="center", foreground_color=(1, 0, 0, 1))
            input_box_row2.add_widget(self.second_input)

        button_box = BoxLayout(orientation='horizontal', spacing=10, size_hint=(0.95, None), height=180, pos_hint={
            'center_x': 0.5, 'center_y': 0.5})
        button_box_1 = Button(text='Central Limit Theorem\n(CLT) Demo', font_size=40, halign='center', size_hint=(0.5,
                                                                                                                 1),
                              pos_hint={'center_y': 0.9})
        global button_box_2
        global button_box_3
        button_box_2 = Button(text='Explanation\nAudio 1', font_size=40, halign='center', size_hint=(0.25, 1),
                              pos_hint={'center_y': 0.9}, on_press=self.play_audio_1)
        button_box_3 = Button(text='Explanation\nAudio 2', font_size=40, halign='center', size_hint=(0.25, 1),
                              pos_hint={'center_y': 0.9}, on_press=self.play_audio_2)

        button_box.add_widget(button_box_1)
        button_box.add_widget(button_box_2)
        button_box.add_widget(button_box_3)

        # Create Plot button
        button = Button(text='Create Plot\n (Click again to regenerate the simulations)', size_hint=(0.95, 0.25),
                        font_size=50, halign = 'center',
                        pos_hint={'center_x': 0.5})
        button.bind(on_release=self.create_plot)

        input_box.add_widget(self.N_input)
        input_box.add_widget(input_box_row2)

        self.box.add_widget(button_box)
        self.box.add_widget(input_box)
        self.box.add_widget(button)

        self.add_widget(self.box)

    def play_audio_1(self, instance):
        if audio_1.state != 'play':
            audio_1.play()
            button_box_3.disabled = True
            button_box_2.text = 'Stop\nAudio'
            audio_2.stop()
        else:
            audio_1.stop()
            button_box_3.disabled = False
            button_box_2.text = 'Explanation\nAudio 1'

    def play_audio_2(self, instance):
        if audio_2.state != 'play':
            audio_2.play()
            button_box_2.disabled = True
            button_box_3.text = 'Stop\nAudio'
            audio_1.stop()
        else:
            audio_2.stop()
            button_box_2.disabled = False
            button_box_3.text = 'Explanation\nAudio 2'

    def show_popup(self, instance):
        popup_x_size = 400 * 2.2
        popup_y_size = 450 * 2.2
        # Create a label with the message
        message_label = Label(text='', size_hint=(1, None), pos_hint={'center_y': 0.6}, halign='center')

        if current_type == 1:
            message_label.text = 'A normal (or Gaussian; after Carl Friedrich Gauss) distribution is a continuous ' \
                                 'probability distribution for a real-valued random variable. Normal distribution ' \
                                 'has two parameters, which are the mean and the standard deviation, respectively. ' \
                                 'It is the most important distribution in statistical data analysis.  [' \
                                 'ref=https://en.wikipedia.org/wiki/Normal_distribution]'

        elif current_type == 2:
            message_label.text = 'The Poisson (Siméon Denis Poisson) distribution is a discrete probability ' \
                                 'distribution that describes the probability of a given number of events occurring ' \
                                 'in a fixed interval of time. The Poisson distribution has only one parameter, ' \
                                 'which is the rate. The Poisson distribution is often used to model rare events. ' \
                                 'The parameter of a Poisson distribution represents both its expectation (mean) and ' \
                                 'variance.'

        elif current_type == 3:
            message_label.text = 'The binomial distribution is the discrete probability distribution of the number ' \
                                 'of successes in a sequence of n independent experiments. For a single experiment, ' \
                                 'i.e., n = 1, the binomial distribution is a Bernoulli (Jacob Bernoulli) ' \
                                 'distribution. The binomial distribution has two parameters, including n (number of ' \
                                 ' experiments) and p (success rate).'

        elif current_type == 4:
            message_label.text = 'The beta distribution is a family of continuous probability distributions defined ' \
                                 'on the interval of 0 and 1. The beta distribution has two parameters, denoted by ' \
                                 'alpha (α) and beta (β), respectively, which control the shape of the distribution. ' \
                                 'It is widely used as a prior distribution in Bayesian data analysis.'

        elif current_type == 5:
            message_label.text = 'The negative binomial distribution is a discrete probability distribution that ' \
                                 'models the number of failures in a sequence of independent and identically ' \
                                 'distributed Bernoulli trials before a specified number of successes occurs. The ' \
                                 'negative binomial distribution has two parameters, which are the number of ' \
                                 'successes required (r) and the probability of success in a single trial (p). The ' \
                                 'negative binomial distribution offers more flexibility than the Poisson ' \
                                 'distribution for modeling count or event data because it does not require the mean ' \
                                 'to be equal to the variance.'

        elif current_type == 6:
            message_label.text = 'The continuous uniform distribution is a symmetric probability distribution that ' \
                                 'describes an experiment where there is an equally likely outcome within a certain ' \
                                 'lower and upper bounds of the interval. It is often used as a non-informative ' \
                                 'prior in Bayesian data analysis, as it does not impose any prior knowledge. ' \
                                 'However, it is still informative in the sense that its mean is the average of ' \
                                 'lower and upper bounds.'

        elif current_type == 7:
            message_label.text = 'The chi-squared distribution with k degrees of freedom is the distribution of a sum' \
                                 'sum of the squares of k independent standard normal random variables. The ' \
                                 'chi-squared distribution is a special case of the gamma distribution and is one of ' \
                                 'the most widely used probability distributions in inferential statistics.'

        elif current_type == 8:
            message_label.text = 'The exponential distribution is the probability distribution of the time between ' \
                                 'events in a Poisson point process. It is a special case of the gamma distribution, ' \
                                 'characterized by the rate parameter. The exponential distribution is often ' \
                                 'described as memory-less because the probability of an event occurring within a ' \
                                 'certain time interval does not depend on the elapsed time. '

        elif current_type == 9:
            message_label.text = 'Student (named after William Sealy Gosset) t-distribution is a continuous ' \
                                 'probability distribution that generalizes the standard normal distribution. For a ' \
                                 'degree of freedom equal to 1, the Student t distribution becomes the standard ' \
                                 'Cauchy distribution, whereas for degrees of freedom approaching infinity, ' \
                                 'the t-distribution converges to the standard normal distribution. The two-sample ' \
                                 't-test is commonly used and is based on the t-distribution. It allows for unequal ' \
                                 'variances between the two groups being compared. '

        message_label.bind(
            width=lambda *x: message_label.setter('text_size')(message_label, (message_label.width,
                                                                               None)))

        # Create the popup and set its content
        popup = Popup(title='Description', content=message_label, size_hint=(None, None), size=(popup_x_size,
                                                                                                popup_y_size))
        # Open the popup
        popup.open()

    def show_home_screen(self, instance):
        self.manager.current = 'home_page'

    def hook_keyboard(self, window, key, *largs):
        if key == 27:
            # do what you want, return True for stopping the propagation
            self.show_home_screen(1)
            return True

    # noinspection PyGlobalUndefined
    def create_plot(self, instance):
        global current_type
        global current_type_name
        global needed_inputs
        global first_input_name
        global second_input_name
        global first_input_needs_int
        global second_input_needs_int
        first_value = None
        second_value = None
        data = None
        self.errors.text = ''

        if not first_input_needs_int:
            try:
                first_value = float(self.first_input.text)
            except ValueError:
                self.errors.text = f'{first_input_name} must be a number.'
                return
        elif first_input_needs_int:
            try:
                first_value = int(self.first_input.text)
            except ValueError:
                self.errors.text = f'{first_input_name} must be an integer.'
                return
        if needed_inputs >= 2:
            if not second_input_needs_int:
                try:
                    second_value = float(self.second_input.text)
                except ValueError:
                    self.errors.text = f'{second_input_name} must be a number.'
                    return
            elif second_input_needs_int:
                try:
                    second_value = int(self.second_input.text)
                except ValueError:
                    self.errors.text = f'{second_input_name} must be an integer.'
                    return

        if self.N_input.text != 'Number of Simulations\n(default is 10000). Click to change':
            try:
                size = int(self.N_input.text) if self.N_input.text else None
            except ValueError:
                self.errors.text = 'Number of iterations (N) must be an integer.'
                return
        else:
            size = 10000

        # print(size, first_value, second_value, current_type)

        if current_type == 1:
            data = np.random.normal(first_value, second_value, size=size)
        elif current_type == 2:
            data = np.random.poisson(first_value, size=size)
        elif current_type == 3:
            data = np.random.binomial(first_value, second_value, size=size)
        elif current_type == 4:
            data = np.random.beta(first_value, second_value, size=size)
        elif current_type == 5:
            data = np.random.negative_binomial(first_value, second_value, size=size)
        elif current_type == 6:
            data = np.random.uniform(first_value, second_value, size=size)
        elif current_type == 7:
            data = np.random.chisquare(first_value, size=size)
        elif current_type == 8:
            data = np.random.exponential(first_value, size=size)
        elif current_type == 9:
            data = np.random.standard_t(first_value, size=size)

        # Clear the existing plot
        for child in self.box.children[:]:
            if isinstance(child, FigureCanvasKivyAgg):
                self.box.remove_widget(child)

        fig, axs = plt.subplots(2, 1)
        mpl.rcParams['font.size'] = 16
        label_font_size = 16

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
        self.errors.text = ''

    def go_to_start_screen(self, instance, touch):
        if touch.is_double_tap:
            self.box.clear_widgets()  # Clear all widgets in the box layout
            self.manager.current = 'home_page'

    def on_leave(self):
        # Clear previous content
        self.clear_widgets()

        # Remove all cached screens
        # self.manager.clear_widgets()
        global current_type
        current_type = 0


class MyApp(App):
    def build(self):
        self.screen_manager = ScreenManager()

        home_page = HomePage(name='home_page')
        self.screen_manager.add_widget(home_page)

        first_page = FirstPage(name='first_page')
        self.screen_manager.add_widget(first_page)

        return self.screen_manager


if __name__ == '__main__':
    MyApp().run()
