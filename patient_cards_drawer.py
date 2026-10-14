"""Slide-out patient cards drawer with password protection and graph visualizations."""

import math
from PyQt6.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                            QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget)


class MiniPatientGraphWidget(QWidget):
    def __init__(self, topology=None, parent=None):
        super().__init__(parent)
        self.topology = topology or {'nodes': [], 'edges': []}
        self.setFixedHeight(95)
        self.setMinimumWidth(320)
        self.setStyleSheet('background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 0px;')

    def set_topology(self, topology):
        self.topology = topology or {'nodes': [], 'edges': []}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor('#F8FAFC'))
        painter.setPen(QPen(QColor('#E2E8F0'), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        nodes = self.topology.get('nodes', [])
        edges = self.topology.get('edges', [])

        patients = [n for n in nodes if n.get('type') == 'PATIENT']
        sessions = [n for n in nodes if n.get('type') == 'SESSION']
        biomarkers = [n for n in nodes if n.get('type') == 'BIOMARKER']
        syndromes = [n for n in nodes if n.get('type') == 'SYNDROME']

        if not sessions:
            painter.setFont(QFont('Consolas', 8))
            painter.setPen(QColor('#64748B'))
            painter.drawText(QRect(10, 10, w - 20, h - 20),
                             Qt.AlignmentFlag.AlignCenter,
                             'Граф пуст: нет сохраненных сессий')
            return

        cols = [
            {'name': 'ПАЦ', 'x': 25, 'color': '#0F172A', 'items': patients[:1]},
            {'name': 'СЕС', 'x': 105, 'color': '#2563EB', 'items': sessions[:3]},
            {'name': 'БИО', 'x': 195, 'color': '#64748B', 'items': biomarkers[:3]},
            {'name': 'СИН', 'x': 285, 'color': '#0F172A', 'items': syndromes[:3]},
        ]

        node_positions = {}

        for col in cols:
            items = col['items']
            cx = col['x']
            count = len(items)
            for idx, item in enumerate(items):
                cy = h // 2 if count == 1 else int(20 + (h - 40) * idx / max(1, count - 1))
                nid = item.get('id')
                node_positions[nid] = (cx, cy)

                painter.setPen(QPen(QColor(col['color']), 1.2))
                painter.setBrush(QBrush(QColor('#FFFFFF')))
                bw, bh = 46, 16
                painter.drawRect(cx - bw // 2, cy - bh // 2, bw, bh)

                painter.setFont(QFont('Consolas', 7, QFont.Weight.Bold))
                painter.setPen(QColor('#0F172A'))
                label = item.get('label', '')
                if item.get('type') == 'SESSION':
                    display = f"С#{item.get('session_index', idx + 1)}"
                elif item.get('type') == 'PATIENT':
                    display = "ПАЦ"
                elif item.get('type') == 'BIOMARKER':
                    display = label[:5] + '…' if len(label) > 6 else label
                else:
                    display = label[:5] + '…' if len(label) > 6 else label

                painter.drawText(QRect(cx - bw // 2, cy - bh // 2, bw, bh),
                                 Qt.AlignmentFlag.AlignCenter, display)

        for edge in edges:
            u = edge.get('from')
            v = edge.get('to')
            etype = edge.get('type')
            trend = edge.get('trend')
            if u in node_positions and v in node_positions:
                p1 = node_positions[u]
                p2 = node_positions[v]
                if etype == 'NEXT_SESSION':
                    color = '#10B981' if trend == 'УЛУЧШЕНИЕ' else '#DC2626' if trend == 'УХУДШЕНИЕ' else '#64748B'
                    pen = QPen(QColor(color), 1.5)
                elif etype in ('HAS_SESSION', 'EXHIBITS'):
                    pen = QPen(QColor('#2563EB'), 1.0)
                else:
                    pen = QPen(QColor('#CBD5E1'), 1.0)
                painter.setPen(pen)
                painter.drawLine(p1[0] + 23, p1[1], p2[0] - 23, p2[1])


class NewPatientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Новая карта пациента')
        self.setFixedSize(420, 395)
        self.setModal(True)
        self.setStyleSheet('''
            QDialog { background: #FFFFFF; border: 1px solid #0F172A; border-radius: 0px; }
            QLabel { color: #0F172A; font-family: 'Segoe UI'; font-size: 8.5pt; font-weight: 500; }
            QLineEdit {
                background: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                padding: 4px 8px;
                border-radius: 0px;
                font-family: 'Segoe UI';
                font-size: 9.5pt;
                min-height: 28px;
            }
            QLineEdit:focus { border-color: #2563EB; }
            QPushButton {
                background: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                padding: 6px 14px;
                border-radius: 0px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover { border-color: #2563EB; color: #2563EB; }
            QPushButton#primaryBtn { background: #0F172A; color: #FFFFFF; border: 1px solid #0F172A; }
            QPushButton#primaryBtn:hover { background: #1E293B; }
        ''')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(6)

        title = QLabel('РЕГИСТРАЦИЯ ПАЦИЕНТА В ГРАФЕ')
        title.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        title.setStyleSheet('color: #0F172A; padding-bottom: 4px;')
        lay.addWidget(title)

        lay.addWidget(QLabel('Фамилия:'))
        self.last_name_input = QLineEdit()
        self.last_name_input.setFixedHeight(30)
        self.last_name_input.setPlaceholderText('Иванов')
        lay.addWidget(self.last_name_input)

        lay.addWidget(QLabel('Имя:'))
        self.first_name_input = QLineEdit()
        self.first_name_input.setFixedHeight(30)
        self.first_name_input.setPlaceholderText('Иван')
        lay.addWidget(self.first_name_input)

        lay.addWidget(QLabel('Пароль доступа (опционально):'))
        self.password_input = QLineEdit()
        self.password_input.setFixedHeight(30)
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText('Введите пароль')
        lay.addWidget(self.password_input)

        lay.addWidget(QLabel('Подтверждение пароля:'))
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setFixedHeight(30)
        self.confirm_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_password_input.setPlaceholderText('Повторите пароль')
        lay.addWidget(self.confirm_password_input)

        self.error_label = QLabel('')
        self.error_label.setStyleSheet('color: #DC2626; font-size: 8pt; min-height: 16px;')
        lay.addWidget(self.error_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.cancel_button = QPushButton('Отмена')
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_button)

        self.create_button = QPushButton('Создать карту')
        self.create_button.setObjectName('primaryBtn')
        self.create_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.create_button.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(self.create_button)
        lay.addLayout(btn_row)

    def _validate_and_accept(self):
        first = self.first_name_input.text().strip()
        last = self.last_name_input.text().strip()
        pwd = self.password_input.text()
        cpwd = self.confirm_password_input.text()

        if not first or not last:
            self.error_label.setText('Укажите фамилию и имя пациента.')
            return
        if pwd != cpwd:
            self.error_label.setText('Пароли не совпадают.')
            return
        self.accept()

    def get_data(self):
        return (
            self.first_name_input.text().strip(),
            self.last_name_input.text().strip(),
            self.password_input.text() or None
        )


class PasswordPromptDialog(QDialog):
    def __init__(self, patient_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Авторизация доступа')
        self.setFixedSize(380, 200)
        self.setModal(True)
        self.setStyleSheet('''
            QDialog { background: #FFFFFF; border: 1px solid #0F172A; border-radius: 0px; }
            QLabel { color: #0F172A; font-family: 'Segoe UI'; font-size: 8.5pt; }
            QLineEdit {
                background: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                padding: 4px 8px;
                border-radius: 0px;
                font-family: 'Segoe UI';
                font-size: 9.5pt;
                min-height: 28px;
            }
            QLineEdit:focus { border-color: #2563EB; }
            QPushButton {
                background: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                padding: 6px 14px;
                border-radius: 0px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover { border-color: #2563EB; }
            QPushButton#primaryBtn { background: #0F172A; color: #FFFFFF; border: 1px solid #0F172A; }
            QPushButton#primaryBtn:hover { background: #1E293B; }
        ''')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        title = QLabel(f'КАРТА: {patient_label}')
        title.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        lay.addWidget(title)

        lay.addWidget(QLabel('Введите пароль доступа к графу:'))
        self.password_input = QLineEdit()
        self.password_input.setFixedHeight(30)
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        lay.addWidget(self.password_input)

        self.error_label = QLabel('')
        self.error_label.setStyleSheet('color: #DC2626; font-size: 8pt; min-height: 16px;')
        lay.addWidget(self.error_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.cancel_button = QPushButton('Отмена')
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_button)

        self.ok_button = QPushButton('Разблокировать')
        self.ok_button.setObjectName('primaryBtn')
        self.ok_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_button.clicked.connect(self.accept)
        btn_row.addWidget(self.ok_button)
        lay.addLayout(btn_row)

    def get_password(self):
        return self.password_input.text()

    def set_error(self, message):
        self.error_label.setText(message)


class PatientGraphCard(QFrame):
    activated = pyqtSignal(str)
    show_full_graph_requested = pyqtSignal(str)
    unlocked_changed = pyqtSignal(str, bool)

    def __init__(self, patient_id, patient_graph, parent=None, is_active=False):
        super().__init__(parent)
        self.patient_id = patient_id
        self.patient_graph = patient_graph
        self.is_active = is_active
        self.is_unlocked = not patient_graph.has_password(patient_id)
        self.setObjectName('patientCard')
        self.setStyleSheet('''
            QFrame#patientCard { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 0px; }
            QLabel { color: #0F172A; background: transparent; border: none; font-family: 'Segoe UI'; }
            QPushButton { background: #FFFFFF; color: #1D4ED8; border: 1px solid #CBD5E1; padding: 4px 8px; border-radius: 0px; font-family: 'Segoe UI'; font-size: 8pt; font-weight: bold; }
            QPushButton:hover { border-color: #2563EB; background: #F8FAFC; }
        ''')

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 8, 10, 8)
        self.layout.setSpacing(5)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        self.status_badge = QLabel()
        header_row.addWidget(self.status_badge)

        self.name_label = QLabel()
        self.name_label.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.name_label.setStyleSheet('color: #0F172A;')
        header_row.addWidget(self.name_label, 1)

        self.header_label = self.name_label

        self.active_badge = QLabel('АКТИВЕН')
        self.active_badge.setStyleSheet('color: #1D4ED8; background: #EFF6FF; border: 1px solid #93C5FD; font-family: \'Segoe UI\'; font-size: 7.5pt; font-weight: bold; padding: 1px 6px; border-radius: 0px;')
        header_row.addWidget(self.active_badge)
        self.layout.addLayout(header_row)

        self.id_lbl = QLabel(f"ID узла: {self.patient_id}")
        self.id_lbl.setFont(QFont('Consolas', 7))
        self.id_lbl.setStyleSheet('color: #64748B;')
        self.layout.addWidget(self.id_lbl)

        self.locked_box = QWidget()
        locked_lay = QVBoxLayout(self.locked_box)
        locked_lay.setContentsMargins(0, 2, 0, 2)
        locked_lay.setSpacing(4)
        self.info_label = QLabel()
        self.info_label.setFont(QFont('Segoe UI', 8))
        self.info_label.setStyleSheet('color: #64748B;')
        locked_lay.addWidget(self.info_label)

        self.unlock_button = QPushButton('Разблокировать и открыть граф')
        self.unlock_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.unlock_button.clicked.connect(self._prompt_unlock)
        locked_lay.addWidget(self.unlock_button)
        self.layout.addWidget(self.locked_box)

        self.unlocked_box = QWidget()
        unlocked_lay = QVBoxLayout(self.unlocked_box)
        unlocked_lay.setContentsMargins(0, 2, 0, 2)
        unlocked_lay.setSpacing(5)

        self.diagnoses_label = QLabel()
        self.diagnoses_label.setFont(QFont('Segoe UI', 8))
        self.diagnoses_label.setStyleSheet('color: #0F172A;')
        self.diagnoses_label.setWordWrap(True)
        unlocked_lay.addWidget(self.diagnoses_label)

        self.biomarkers_label = QLabel()
        self.biomarkers_label.setFont(QFont('Segoe UI', 8))
        self.biomarkers_label.setStyleSheet('color: #475569;')
        self.biomarkers_label.setWordWrap(True)
        unlocked_lay.addWidget(self.biomarkers_label)

        self.sessions_label = QLabel()
        self.sessions_label.setFont(QFont('Consolas', 7))
        self.sessions_label.setStyleSheet('color: #334155;')
        self.sessions_label.setWordWrap(True)
        unlocked_lay.addWidget(self.sessions_label)

        self.mini_graph = MiniPatientGraphWidget()
        unlocked_lay.addWidget(self.mini_graph)

        actions = QVBoxLayout()
        actions.setSpacing(4)
        self.activate_button = QPushButton('Сделать активным / новый замер')
        self.activate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.activate_button.clicked.connect(lambda: self.activated.emit(self.patient_id))
        actions.addWidget(self.activate_button)

        self.show_dashboard_button = QPushButton('Показать полный граф на дашборде')
        self.show_dashboard_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_dashboard_button.clicked.connect(lambda: self.show_full_graph_requested.emit(self.patient_id))
        actions.addWidget(self.show_dashboard_button)

        self.lock_button = QPushButton('Заблокировать карту')
        self.lock_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lock_button.clicked.connect(self.lock)
        actions.addWidget(self.lock_button)
        unlocked_lay.addLayout(actions)

        self.layout.addWidget(self.unlocked_box)

        self.refresh()

    def refresh(self):
        summary = self.patient_graph.get_patient_card_subgraph_summary(self.patient_id)
        if not summary:
            return

        label = summary['label']
        sessions = summary['sessions']
        s_count = len(sessions)

        self.name_label.setText(label)
        self.active_badge.setVisible(self.is_active)

        if self.is_unlocked:
            self.status_badge.setText('ОТКРЫТО')
            self.status_badge.setStyleSheet('color: #15803D; background: #DCFCE7; border: 1px solid #86EFAC; font-family: \'Segoe UI\'; font-size: 7.5pt; font-weight: bold; padding: 1px 6px; border-radius: 0px;')
            self.locked_box.hide()
            self.unlocked_box.show()

            diagnoses = summary['all_diagnoses']
            diag_text = 'Сводка диагнозов / синдромов (из узлов SYNDROME):\n'
            if diagnoses:
                diag_text += '\n'.join(f"• {d['label']}: {d['score']:.0f}%" for d in diagnoses[:3])
            else:
                diag_text += '• Нет зарегистрированных синдромов'
            self.diagnoses_label.setText(diag_text)

            bios = summary['all_biomarkers']
            bio_text = 'Выявленные биомаркеры (из узлов BIOMARKER):\n'
            if bios:
                bio_text += '\n'.join(f"• {b['label']}" for b in bios[:3])
            else:
                bio_text += '• Нет активных биомаркеров'
            self.biomarkers_label.setText(bio_text)

            sess_text = 'История сессий и динамика (Δ):\n'
            if sessions:
                sess_lines = []
                for s in sessions[-3:]:
                    ts = s['timestamp'][:16].replace('T', ' ')
                    trend = s.get('trend', 'СТАБИЛЬНО')
                    delta = s.get('delta_summary', '')
                    sess_lines.append(f"• Сессия #{s['session_index']} ({ts}) — {trend}")
                sess_text += '\n'.join(sess_lines)
            else:
                sess_text += '• Сессии не зафиксированы'
            self.sessions_label.setText(sess_text)

            self.mini_graph.set_topology(summary['graph_topology'])
            self.lock_button.setVisible(summary['is_password_protected'])
        else:
            self.status_badge.setText('ЗАКРЫТО')
            self.status_badge.setStyleSheet('color: #B91C1C; background: #FEE2E2; border: 1px solid #FCA5A5; font-family: \'Segoe UI\'; font-size: 7.5pt; font-weight: bold; padding: 1px 6px; border-radius: 0px;')
            self.info_label.setText(f"Узлов сессий в графе: {s_count} | Данные и диагнозы скрыты паролем")
            self.unlocked_box.hide()
            self.locked_box.show()

    def _prompt_unlock(self):
        summary = self.patient_graph.get_patient_card_subgraph_summary(self.patient_id)
        label = summary['label'] if summary else self.patient_id
        dlg = PasswordPromptDialog(label, self)
        while dlg.exec():
            pwd = dlg.get_password()
            if self.unlock(pwd):
                break
            else:
                dlg.set_error('Неверный пароль доступа к графу пациента')

    def unlock(self, password):
        if self.patient_graph.verify_password(self.patient_id, password):
            self.set_unlocked(True)
            return True
        return False

    def lock(self):
        self.set_unlocked(False)

    def set_unlocked(self, unlocked):
        self.is_unlocked = unlocked
        self.refresh()
        self.unlocked_changed.emit(self.patient_id, unlocked)

    def set_active(self, active):
        self.is_active = active
        self.active_badge.setVisible(active)


class PatientCardsDrawer(QFrame):
    def __init__(self, parent=None, patient_graph=None, session=None, cdss=None):
        super().__init__(parent)
        self.patient_graph = patient_graph
        self.session = session
        self.cdss = cdss
        self.is_open = False
        self.cards = {}
        self.unlocked_patients = set()

        self.setFixedWidth(420)
        self.setStyleSheet('''
            PatientCardsDrawer {
                background: #FFFFFF;
                border-left: 2px solid #0F172A;
                border-top: none;
                border-right: none;
                border-bottom: none;
                border-radius: 0px;
            }
            QLabel { color: #0F172A; background: transparent; font-family: 'Segoe UI'; }
            QScrollArea { background: #FFFFFF; border: none; }
            QPushButton {
                background: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                padding: 5px 12px;
                border-radius: 0px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: bold;
            }
            QPushButton:hover { border-color: #2563EB; color: #2563EB; background: #F8FAFC; }
            QPushButton#actionBtn { background: #0F172A; color: #FFFFFF; border: 1px solid #0F172A; }
            QPushButton#actionBtn:hover { background: #1E293B; }
        ''')

        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(14, 12, 14, 12)
        main_lay.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        title = QLabel('ГРАФОВАЯ КАРТОТЕКА ПАЦИЕНТОВ')
        title.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        header_row.addWidget(title, 1)

        self.close_button = QPushButton('✕')
        self.close_button.setFixedSize(28, 28)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.setStyleSheet('''
            QPushButton {
                background: #F1F5F9;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                padding: 0px;
                margin: 0px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                font-weight: bold;
                text-align: center;
                border-radius: 0px;
            }
            QPushButton:hover {
                background: #EF4444;
                color: #FFFFFF;
                border-color: #EF4444;
            }
            QPushButton:pressed {
                background: #DC2626;
                color: #FFFFFF;
                border-color: #DC2626;
            }
        ''')
        self.close_button.clicked.connect(self.close_drawer)
        header_row.addWidget(self.close_button)
        main_lay.addLayout(header_row)

        self.stats_badge = QLabel('Загрузка графа…')
        self.stats_badge.setFont(QFont('Consolas', 8))
        self.stats_badge.setStyleSheet('color: #64748B; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;')
        main_lay.addWidget(self.stats_badge)

        self.new_patient_dialog_button = QPushButton('+ Новая карта пациента')
        self.new_patient_dialog_button.setObjectName('actionBtn')
        self.new_patient_dialog_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_patient_dialog_button.clicked.connect(self._open_new_patient_dialog)
        main_lay.addWidget(self.new_patient_dialog_button)

        self.save_current_tests_button = QPushButton('Сохранить текущие тесты в карту')
        self.save_current_tests_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_current_tests_button.clicked.connect(self._save_tests_to_active)
        main_lay.addWidget(self.save_current_tests_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet('background: #FFFFFF;')
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 4, 0, 4)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch(1)
        self.scroll_area.setWidget(self.cards_container)
        main_lay.addWidget(self.scroll_area, 1)

        if self.patient_graph:
            self.refresh_cards()

    def update_geometry(self):
        if not self.parent():
            return
        pw = self.parent().width()
        ph = self.parent().height()
        nav_h = self.parent().nav.height() if hasattr(self.parent(), 'nav') else 36
        status_h = self.parent().status_bar.height() if hasattr(self.parent(), 'status_bar') else 22
        y = nav_h
        h = ph - nav_h - status_h
        x = pw - 420 if self.is_open else pw
        self.setGeometry(x, y, 420, h)

    def open_drawer(self):
        self.is_open = True
        self.show()
        self.raise_()
        self.refresh_cards()
        if not self.parent():
            return
        pw = self.parent().width()
        ph = self.parent().height()
        nav_h = self.parent().nav.height() if hasattr(self.parent(), 'nav') else 36
        status_h = self.parent().status_bar.height() if hasattr(self.parent(), 'status_bar') else 22
        y = nav_h
        h = ph - nav_h - status_h
        self.anim.stop()
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(QRect(pw - 420, y, 420, h))
        self.anim.start()
        if hasattr(self.parent(), 'patient_drawer_button'):
            self.parent().patient_drawer_button.setChecked(True)

    def close_drawer(self):
        self.is_open = False
        if not self.parent():
            self.hide()
            return
        pw = self.parent().width()
        ph = self.parent().height()
        nav_h = self.parent().nav.height() if hasattr(self.parent(), 'nav') else 36
        status_h = self.parent().status_bar.height() if hasattr(self.parent(), 'status_bar') else 22
        y = nav_h
        h = ph - nav_h - status_h
        self.anim.stop()
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(QRect(pw, y, 420, h))
        self.anim.start()
        if hasattr(self.parent(), 'patient_drawer_button'):
            self.parent().patient_drawer_button.setChecked(False)

    def toggle(self):
        if self.is_open:
            self.close_drawer()
        else:
            self.open_drawer()

    def refresh_cards(self):
        if not self.patient_graph:
            return
        patients = self.patient_graph.list_patients()
        total_nodes = self.patient_graph.graph.number_of_nodes()
        total_sessions = self.patient_graph.session_count()
        self.stats_badge.setText(f"Пациентов: {len(patients)} | Сессий: {total_sessions} | Узлов графа: {total_nodes}")

        active_id = self.cdss.patient_combo.currentData() if (self.cdss and hasattr(self.cdss, 'patient_combo')) else None

        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        self.cards.clear()
        for p in patients:
            pid = p['id']
            card = PatientGraphCard(pid, self.patient_graph, self.cards_container, is_active=(pid == active_id))
            if pid in self.unlocked_patients:
                card.is_unlocked = True
                card.refresh()
            card.activated.connect(self._on_card_activated)
            card.show_full_graph_requested.connect(self._on_card_show_full_graph)
            card.unlocked_changed.connect(self._on_card_unlocked_changed)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
            self.cards[pid] = card

    def _open_new_patient_dialog(self):
        dlg = NewPatientDialog(self)
        if dlg.exec():
            first, last, pwd = dlg.get_data()
            pid = self.patient_graph.add_patient(first_name=first, last_name=last, password=pwd)
            self.unlocked_patients.add(pid)
            if self.cdss:
                self.cdss._populate_patients()
                for i in range(self.cdss.patient_combo.count()):
                    if self.cdss.patient_combo.itemData(i) == pid:
                        self.cdss.patient_combo.setCurrentIndex(i)
                        break
            if self.session:
                win = self.window()
                if hasattr(win, 'reset_all_modules'):
                    win.reset_all_modules()
                else:
                    self.session.reset()
            self.refresh_cards()
            self.sync_active_patient(pid)

    def _save_tests_to_active(self):
        if self.cdss:
            self.cdss._on_save_session_clicked()
            self.refresh_cards()

    def _on_card_activated(self, patient_id):
        if self.cdss:
            for i in range(self.cdss.patient_combo.count()):
                if self.cdss.patient_combo.itemData(i) == patient_id:
                    self.cdss.patient_combo.setCurrentIndex(i)
                    break
        if self.session:
            win = self.window()
            if hasattr(win, 'reset_all_modules'):
                win.reset_all_modules()
            else:
                self.session.reset()
        self.sync_active_patient(patient_id)

    def _on_card_show_full_graph(self, patient_id):
        if self.cdss:
            for i in range(self.cdss.patient_combo.count()):
                if self.cdss.patient_combo.itemData(i) == patient_id:
                    self.cdss.patient_combo.setCurrentIndex(i)
                    break
            self.cdss.set_graph_mode('HISTORY')
        win = self.window()
        if hasattr(win, 'switch_page'):
            win.switch_page(0)
            if hasattr(win, 'nav'):
                win.nav.set_active(0)
        self.sync_active_patient(patient_id)

    def _on_card_unlocked_changed(self, patient_id, unlocked):
        if unlocked:
            self.unlocked_patients.add(patient_id)
        else:
            self.unlocked_patients.discard(patient_id)
        if self.cdss and self.cdss.graph_mode == 'HISTORY':
            self.cdss._render_history_graph()

    def sync_active_patient(self, active_patient_id):
        for pid, card in self.cards.items():
            card.set_active(pid == active_patient_id)
