import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest

from PyQt6.QtCore import QAbstractAnimation
from PyQt6.QtWidgets import QApplication

from app import MainWindow


class ReportAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.resize(1200, 750)
        self.window.show()
        self.qt.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.processEvents()

    def pages(self):
        w = self.window
        return [(1, w.tremor_page, w.tremor_page.video_label),
                (2, w.facial_page, w.facial_page.video),
                (3, w.rppg_page, w.rppg_page.video),
                (5, w.voice_page, w.voice_page.chart_area),
                (6, w.motor_page, w.motor_page.video)]

    def test_all_report_viewports_expand_progressively_and_finish_at_half_height(self):
        for index, page, viewport in self.pages():
            with self.subTest(page=index):
                self.window.switch_page(index)
                self.qt.processEvents()
                full = viewport.height()
                panel = page.protocol_panel
                panel.status_label.setText('ЗАПИСЬ ЗАВЕРШЕНА')
                panel.reveal()
                controller = panel.reveal_controller
                self.assertEqual(controller.animation.duration(), 240)
                self.assertEqual(controller.animation.state(), QAbstractAnimation.State.Running)
                controller.animation.setCurrentTime(60)
                self.qt.processEvents()
                first_height = panel.height()
                controller.animation.setCurrentTime(120)
                self.qt.processEvents()
                self.assertGreater(panel.height(), first_height)
                self.assertLess(panel.height(), viewport.height())
                controller.animation.setCurrentTime(240)
                self.qt.processEvents()
                self.assertAlmostEqual(viewport.height(), panel.height(), delta=2)
                self.assertLess(viewport.geometry().bottom(), panel.geometry().top())
                panel.hide()
                self.qt.processEvents()
                self.assertGreaterEqual(viewport.height(), full - 2)

    def test_clear_interrupts_animation_and_new_report_restarts_from_zero(self):
        for index, page, viewport in self.pages():
            with self.subTest(page=index):
                self.window.switch_page(index)
                self.qt.processEvents()
                full = viewport.height()
                panel = page.protocol_panel
                panel.reveal()
                animation = panel.reveal_controller.animation
                animation.setCurrentTime(100)
                panel.clear()
                self.qt.processEvents()
                self.assertTrue(panel.isHidden())
                self.assertEqual(animation.state(), QAbstractAnimation.State.Stopped)
                self.assertGreaterEqual(viewport.height(), full - 2)
                animation.setCurrentTime(240)
                self.assertTrue(panel.isHidden())
                panel.reveal()
                self.assertEqual(panel.reveal_controller.fraction, 0)
                animation.setCurrentTime(240)
                self.qt.processEvents()
                self.assertAlmostEqual(viewport.height(), panel.height(), delta=2)
                panel.hide()

    def test_resizing_and_navigation_preserve_completed_reports(self):
        for index, page, viewport in self.pages():
            with self.subTest(page=index):
                self.window.switch_page(index)
                self.qt.processEvents()
                panel = page.protocol_panel
                panel.reveal()
                panel.reveal_controller.animation.setCurrentTime(90)
                self.window.switch_page(0)
                self.qt.processEvents()
                self.assertFalse(panel.isHidden())
                self.assertEqual(panel.reveal_controller.fraction, .5)
                self.window.resize(1400, 860)
                self.window.switch_page(index)
                self.qt.processEvents()
                self.assertAlmostEqual(viewport.height(), panel.height(), delta=2)
                if not panel.close_button.isHidden():
                    panel.close_button.click()
                else:
                    panel.clear()
                self.qt.processEvents()
                self.assertTrue(panel.isHidden())
                self.window.resize(1200, 750)

    def test_voice_controls_do_not_move_when_charts_split(self):
        page = self.window.voice_page
        self.window.switch_page(5)
        self.qt.processEvents()
        rects = [b.geometry() for b in (page.start, page.upload, page.stop, page.reset)]
        page.protocol_panel.reveal()
        page.protocol_panel.reveal_controller.animation.setCurrentTime(240)
        self.qt.processEvents()
        self.assertEqual(rects, [b.geometry() for b in (page.start, page.upload, page.stop, page.reset)])
        self.assertLess(page.chart_area.geometry().bottom(), page.protocol_panel.geometry().top())

    def test_eye_tracking_has_no_report_animation(self):
        self.assertFalse(hasattr(self.window.aac_page, 'protocol_panel'))


if __name__ == '__main__':
    unittest.main()
