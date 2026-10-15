import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtWidgets import QApplication, QLabel, QTableWidget

from app import MainWindow
from dashboard_catalog import MODULES


class DashboardCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.qt.processEvents()
        self.catalog = self.window.dashboard_page.catalog

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.processEvents()

    def settle(self):
        self.qt.processEvents()
        self.qt.processEvents()

    def test_top_blocks_keep_original_geometry_and_cdss_replaces_architecture(self):
        for width, height, half_width in ((1200, 750, 578), (1400, 860, 678)):
            with self.subTest(width=width):
                self.window.resize(width, height)
                self.settle()
                frames = {label.text(): label.parentWidget() for label in self.window.dashboard_page.findChildren(QLabel)
                          if label.text() in ('СВЕДЕНИЯ О ПЛАТФОРМЕ', 'РЕЕСТР МОДУЛЕЙ', 'АРХИТЕКТУРА')}
                self.assertEqual(frames['СВЕДЕНИЯ О ПЛАТФОРМЕ'].geometry().getRect(), (16, 43, half_width, 255))
                self.assertEqual(frames['РЕЕСТР МОДУЛЕЙ'].geometry().getRect(), (28 + half_width, 43, half_width, 255))
                self.assertNotIn('АРХИТЕКТУРА', frames)
                self.assertEqual(self.window.dashboard_page.cdss.geometry().getRect(), (16, 310, width - 32, 260))

    def test_grid_has_four_research_and_two_auxiliary_cards_with_complete_text(self):
        self.assertEqual(len(self.catalog.cards), 6)
        for card, (kind, title, index, description) in zip(self.catalog.cards, MODULES):
            self.assertEqual(card.title.text(), title)
            self.assertEqual(card.preview.kind, kind)
            self.assertEqual(card.module_index, index)
            self.assertEqual(card.description.text(), description)
            self.assertEqual(card.status.text(), '[ ГОТОВ ]')
            self.assertEqual(card.cursor().shape(), Qt.CursorShape.PointingHandCursor)
        research = self.catalog.cards[:4]
        self.assertEqual(research[0].y(), research[1].y())
        self.assertEqual(research[2].y(), research[3].y())
        self.assertEqual(research[0].x(), research[2].x())
        self.assertEqual(research[1].x(), research[3].x())
        self.assertEqual(self.catalog.cards[4].x(), self.catalog.cards[5].x())
        self.assertLess(self.catalog.cards[4].y(), self.catalog.cards[5].y())

    def test_registry_lists_all_six_modules_with_explicit_readable_colors(self):
        table = self.window.dashboard_page.findChild(QTableWidget)
        self.assertEqual(table.rowCount(), 6)
        self.assertEqual([table.horizontalHeaderItem(i).text() for i in range(3)],
                         ['МОДУЛЬ', 'ТИП', 'СТАТУС'])
        self.assertEqual([table.item(i, 0).text() for i in range(6)],
                         ['ТРЕМОР', 'НЕЙРОМИМИКА', 'ПУЛЬС (rPPG)', 'АЙТРЕКИНГ', 'ГОЛОС', 'МОТОРИКА'])
        for row in range(6):
            self.assertEqual(table.item(row, 2).text(), 'ГОТОВ / ПРОТОТИП')
            for column in range(3):
                self.assertTrue(table.item(row, column).text())
                self.assertEqual(table.item(row, column).foreground().color(), QColor('#475569'))

    def test_click_any_part_of_each_card_switches_matching_tab_without_starting_capture(self):
        for card in self.catalog.cards:
            with self.subTest(card=card.title.text()):
                self.window.switch_page(0)
                self.catalog.ensureWidgetVisible(card)
                self.settle()
                for child in (card.title, card.preview, card.description):
                    QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=child.mapTo(card, child.rect().center()))
                    self.assertEqual(self.window.stack.currentIndex(), card.module_index)
                    self.assertTrue(self.window.nav.buttons[card.module_index].isChecked())
                    self.assertIsNone(self.window.stack.currentWidget().worker)
                    self.window.switch_page(0)
                    self.settle()

    def test_keyboard_activation_and_non_left_click_behavior(self):
        card = self.catalog.cards[3]
        spy = QSignalSpy(card.activated)
        QTest.mouseClick(card, Qt.MouseButton.RightButton)
        QTest.mouseRelease(card, Qt.MouseButton.LeftButton)
        self.assertEqual(len(spy), 0)
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Space):
            QTest.keyClick(card, key)
        self.assertEqual(len(spy), 2)
        self.assertTrue(all(args[0] == 6 for args in spy))

    def test_small_window_scrolls_catalog_without_clipping_card_content(self):
        for width, height in ((1200, 750), (1400, 860)):
            self.window.resize(width, height)
            self.settle()
            self.assertEqual(self.catalog.horizontalScrollBar().maximum(), 0)
            for card in self.catalog.cards:
                self.catalog.ensureWidgetVisible(card)
                self.settle()
                self.assertEqual(card.preview.height(), 76)
                self.assertGreaterEqual(card.description.height(), card.description.heightForWidth(card.description.width()))
                self.assertLess(card.status.geometry().right(), card.width())
                self.assertLessEqual(card.description.geometry().bottom(), card.height() - 8)
            if width == 1200:
                self.assertGreater(self.catalog.verticalScrollBar().maximum(), 0)
            else:
                self.assertGreater(self.catalog.verticalScrollBar().maximum(), 0)

    def test_previews_are_light_vector_drawings_and_hover_does_not_change_geometry(self):
        for card in self.catalog.cards:
            self.catalog.ensureWidgetVisible(card)
            self.settle()
            image = card.preview.grab().toImage()
            slate_background = QColor('#F8FAFC').rgb()
            colored = light = 0
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    light += min(color.red(), color.green(), color.blue()) >= 220
                    colored += color.rgb() != slate_background
            self.assertGreater(colored, image.width() * 2)
            self.assertGreater(light / (image.width() * image.height()), .8)
            before = [c.geometry() for c in self.catalog.cards]
            QTest.mouseMove(card, card.rect().center())
            self.settle()
            self.assertEqual(before, [c.geometry() for c in self.catalog.cards])
            self.assertIn('border: 1px solid #2563EB', card.styleSheet())
            self.assertIn('background: #FFFFFF', card.styleSheet())
            self.assertIn('border-radius: 0px', card.styleSheet())
            self.assertIsNone(card.graphicsEffect())


if __name__ == '__main__':
    unittest.main()
