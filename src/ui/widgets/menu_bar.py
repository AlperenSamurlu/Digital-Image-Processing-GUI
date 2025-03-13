class MenuBar:
    def __init__(self, parent):
        self.parent = parent
        self.menu_bar = self.create_menu_bar()

    def create_menu_bar(self):
        menu_bar = self.parent.menuBar()

        assignment_menu = menu_bar.addMenu("Assignments")

        assignment_1_action = assignment_menu.addAction("Ödev 1: Temel İşlevselliği Oluştur")
        assignment_2_action = assignment_menu.addAction("Ödev 2: Filtre Uygulama")

        assignment_1_action.triggered.connect(self.open_assignment_1)
        assignment_2_action.triggered.connect(self.open_assignment_2)

        return menu_bar

    def open_assignment_1(self):
        pass

    def open_assignment_2(self):
        pass