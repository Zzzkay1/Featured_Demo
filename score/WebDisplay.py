class WebDisplay:
    def __init__(self, data):
        self.data = data

    def print_data(self):
        string = ""
        for key, value in self.data.items():
            print(f"{key}: {value}")
            string += f"{key}: {value}\n"
        #return string

    def render(self):
        # Code to render the web display using self.data
        pass