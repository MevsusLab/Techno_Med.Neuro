import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

from PyQt6.QtWidgets import QApplication
from app import MainWindow
from i18n_ru import ru
from protocol_panel import protocol_text, facial_protocol_text, rppg_protocol_text


class RussianTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_module_titles_and_action_buttons_are_russian(self):
        window = MainWindow()
        try:
            from PyQt6.QtWidgets import QLabel
            for page, heading in ((window.tremor_page, 'АНАЛИЗ ДВИЖЕНИЙ КИСТИ'),
                                  (window.facial_page, 'СИММЕТРИЯ ЛИЦА'),
                                  (window.rppg_page, 'БЕСКОНТАКТНАЯ'),
                                  (window.voice_page, 'ГОЛОСОВЫЕ')):
                self.assertTrue(any(heading in label.text() for label in page.findChildren(QLabel)))
            self.assertEqual(window.tremor_page.btn_start.text(), 'НАЧАТЬ ЗАПИСЬ')
            self.assertEqual(window.facial_page.start.text(), 'НАЧАТЬ ТЕСТ ЛИЦА')
            self.assertEqual(window.voice_page.start.text(), 'ЗАПИСАТЬ ГОЛОС')
            self.assertEqual(window.aac_page.btn_tracking.text(), 'НАЧАТЬ ОТСЛЕЖИВАНИЕ ВЗГЛЯДА')
            self.assertEqual(window.aac_page.btn_calibrate.text(), 'КАЛИБРОВАТЬ ВЗГЛЯД')
            self.assertEqual(window.aac_page.btn_stop_tracking.text(), 'ОСТАНОВИТЬ ОТСЛЕЖИВАНИЕ')
            for button in (window.tremor_page.btn_video, window.facial_page.upload_video,
                           window.rppg_page.upload_video, window.motor_page.upload_video):
                self.assertEqual(button.text(), 'ЗАГРУЗИТЬ ВИДЕОФАЙЛ')
            self.assertEqual(window.voice_page.upload.text(), 'ЗАГРУЗИТЬ АУДИОФАЙЛ')
            self.assertEqual(window.aac_page.gaze_x.findChildren(QLabel)[0].text(), 'ВЗГЛЯД ПО ГОРИЗОНТАЛИ')
            self.assertEqual(window.aac_page.gaze_y.findChildren(QLabel)[0].text(), 'ВЗГЛЯД ПО ВЕРТИКАЛИ')
            self.assertEqual(window.aac_page.dwell_display.findChildren(QLabel)[0].text(), 'УДЕРЖАНИЕ ВЗГЛЯДА')
            self.assertEqual(window.tremor_page.page_badge.text(), 'IDLE')
            self.assertEqual([button.text() for button in window.nav.buttons],
                             ['ОБЗОР', 'ТРЕМОР', 'НЕЙРОМИМИКА', 'ПУЛЬС', 'АЙТРЕКИНГ', 'ГОЛОС', 'МОТОРИКА'])
        finally:
            window.close()

    def test_dynamic_prompts_preserve_measurements(self):
        self.assertIn('Улыбнитесь', ru('Smile clearly and hold / CAPTURING'))
        self.assertIn('Поднимите обе брови', ru('Raise both eyebrows and hold / PREPARE'))
        self.assertIn('покой — 90', ru('Complete. Valid frames: rest: 90 / smile: 90 / brows: 90. Geometric percentages are not clinical risk scores.'))
        self.assertIn('1–4.7 Гц', ru('Limited to 1-4.7 Hz by a 0.085s frame gap. Faster motion cannot be assessed; retry for a wider range. Not a diagnosis.'))
        self.assertIn('0.085', ru('Limited to 1-4.7 Hz by a 0.085s frame gap.'))

    def test_post_test_actions_and_warnings_are_translated_for_all_branches(self):
        for frequency in (None, 1.2, 5, 8, 13):
            content = protocol_text({'dominant_freq': frequency, 'concentration': 80, 'rms_px': 1, 'max_freq': 12})
            for value in content:
                self.assertRegex(value, '[А-Яа-я]')
        for asi, smile, brow in ((2, 5, 2), (10, 15, 5), (17, 30, 5), (17, 30, 20)):
            content = facial_protocol_text({'asymmetry_index': asi, 'smile_imbalance': smile, 'brow_imbalance': brow})
            for value in content:
                self.assertRegex(value, '[А-Яа-я]')
        for bpm, concentration in ((67, 40), (55, 80), (67, 80), (110, 80)):
            content = rppg_protocol_text({'bpm': bpm, 'concentration': concentration, 'coverage': 0.91, 'roi_bpm': [bpm]*3, 'warnings': []})
            for value in content:
                self.assertRegex(value, '[А-Яа-я]')
            self.assertNotIn('Seek', ru(content[2]))
        self.assertIn('Непрерывно тяните', ru('Sustain Ah continuously. Keep volume and microphone distance steady.'))


if __name__ == '__main__':
    unittest.main()
