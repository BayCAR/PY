
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from kivymd.uix.button import MDIconButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


class FoodItem:
    def __init__(self, name, icon_path, expiration_date):
        self.name = name
        self.icon_path = icon_path
        self.expiration_date = expiration_date


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [0, 0, 0, 0] # left, top, right, bottom padding

        with self.canvas.before:
            Color(1, 0, 0, 1)  # Set the background color (in this example, red)
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self.update_rect, pos=self.update_rect)

        # Create a top bar box
        top_bar = BoxLayout(orientation='vertical', size_hint=(1, None), height=dp(40))

        # Create a blue bar
        with top_bar.canvas.before:
            Color(0, 0, 1)  # Set the color to blue
            self.bar_rect = Rectangle(pos=top_bar.pos, size=top_bar.size)

        top_bar.bind(pos=self.update_bar_pos, size=self.update_bar_size)

        self.add_widget(top_bar)

        # Create a scrollable list of items
        scroll_view = ScrollView()
        self.item_list = BoxLayout(orientation='vertical', padding=(0, dp(15), 0, 0), spacing=40, size_hint_y=None)
        self.item_list.bind(minimum_height=self.item_list.setter('height'))
        scroll_view.add_widget(self.item_list)
        self.add_widget(scroll_view)

        # Create a bottom button to add new items
        button_container = BoxLayout(size_hint=(None, None), size=(100, 100), pos_hint={'center_x': 0.5})
        add_button = Button(text='+', font_size=72, size_hint=(None, None), size=(100, 100))
        add_button.bind(on_press=self.open_add_item_popup)
        button_container.add_widget(add_button)
        self.add_widget(button_container)

        # Populate the list with example items
        self.add_example_items()

    def update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_bar_pos(self, instance, pos):
        self.bar_rect.pos = pos

    def update_bar_size(self, instance, size):
        self.bar_rect.size = size

    def delete_item(self, button):
        item = button.parent
        self.item_list.remove_widget(item)

    def add_example_items(self):
        # Add example food items to the list
        example_items = [
            FoodItem("Milk", "milk.png", "2023-06-10"),
            FoodItem("Bread", "bread.png", "2023-06-05"),
            FoodItem("Cheese", "cheese.png", "2023-06-15"),
        ]

        for item in example_items:
            item_widget = BoxLayout(orientation='horizontal', spacing=10)
            item_widget.add_widget(Image(source=item.icon_path))
            item_widget.add_widget(Label(text=item.name))
            item_widget.add_widget(Label(text=item.expiration_date))
            edit_button = Button(text="Edit", size_hint=(None, None), size=(50, 30),
                                 pos_hint={'right': 0.8, 'center_y': 0.5})
            delete_button = Button(text="Delete", size_hint=(None, None), size=(70, 30),
                                   pos_hint={'right': 0.9, 'center_y': 0.5},
                                   on_release=self.delete_item)

            item_widget.add_widget(edit_button)
            item_widget.add_widget(delete_button)
            self.item_list.add_widget(item_widget)

    def open_add_item_popup(self, instance):
        # Create a popup for adding new items
        content = BoxLayout(orientation='vertical', spacing=5, padding=8)
        name_input = TextInput(hint_text='Name Ex: Lettuce')
        expiration_input = TextInput(hint_text='Expiration Date Ex: 2018-04-27')
        add_button = Button(text='Add')
        content.add_widget(name_input)
        content.add_widget(expiration_input)
        content.add_widget(add_button)

        popup = Popup(title='Add New Item', content=content, size_hint=(None, None), size=(400, 200))
        add_button.bind(on_press=lambda instance: self.add_new_item(name_input.text, expiration_input.text, popup))
        popup.open()

    def add_new_item(self, name, expiration_date, popup):
        # Add the new item to the list
        item_widget = BoxLayout(orientation='horizontal', spacing=10)
        item_widget.add_widget(Image(source='default_icon.png'))
        item_widget.add_widget(Label(text=name))
        item_widget.add_widget(Label(text=expiration_date))
        edit_button = Button(text="Edit", size_hint=(None, None), size=(50, 30),
                             pos_hint={'right': 0.8, 'center_y': 0.5})
        delete_button = Button(text="Delete", size_hint=(None, None), size=(70, 30),
                               pos_hint={'right': 0.9, 'center_y': 0.5},
                               on_release=self.delete_item)
        item_widget.add_widget(edit_button)
        item_widget.add_widget(delete_button)
        self.item_list.add_widget(item_widget)
        popup.dismiss()


class FoodManagementApp(App):
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    FoodManagementApp().run()
