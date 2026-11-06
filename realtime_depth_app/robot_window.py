"""Responsive Qt dashboard for deterministic, memory-driven exploration."""
import os
from pathlib import Path
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import PySide6

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(PySide6.__file__).resolve().parent/'Qt/plugins')
from PySide6 import QtCore, QtGui, QtWidgets


from robot_map_view import MapView


def make_table(headers):
    table = QtWidgets.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().hide()
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.setWordWrap(True)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(len(headers)-1, QtWidgets.QHeaderView.Stretch)
    return table


def fill_table(table, rows):
    if table.rowCount() != len(rows):
        table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for col, value in enumerate(values):
            text = str(value)
            item = table.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem(text)
                table.setItem(row, col, item)
            elif item.text() != text:
                item.setText(text)
    table.resizeRowsToContents()


class RobotWindow(QtWidgets.QMainWindow):
    def __init__(self, node, worker, args, save):
        super().__init__()
        self.node, self.worker, self.args, self.save = node, worker, args, save
        self.exports = ThreadPoolExecutor(max_workers=1, thread_name_prefix='robot-export')
        self.export_job = None
        self.last_map_revision = None
        self.last_depth_id = None
        self.last_object_id = None
        self.opened = time.monotonic()
        self.setWindowTitle('JERRY · SLAM exploration and depth')
        self.resize(1320, 820)
        self.setStyleSheet('''
            QMainWindow,QWidget {background:#111b25;color:#e7edf4;font-family:Sans Serif;font-size:13px;}
            QTableWidget {background:#152331;gridline-color:#2a3d4f;}
            QHeaderView::section {background:#23384a;padding:7px;border:0;font-weight:600;}
            QPushButton {background:#253d51;border:1px solid #3d5b72;border-radius:5px;padding:9px 14px;}
            QPushButton:hover {background:#355871;} QPushButton:disabled {color:#90a4b6;}
            QPushButton#start {background:#166e5b;} QPushButton#stop {background:#963747;}
            QTabBar::tab {padding:9px 15px;} QLabel#banner {background:#203449;padding:10px;border-radius:6px;}
        ''')
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        heading = QtWidgets.QLabel('JERRY  /  Curious exploration')
        heading.setStyleSheet('font-size:23px;font-weight:700;')
        layout.addWidget(heading)
        toolbar = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton('Start exploring')
        self.start.setObjectName('start')
        self.start.clicked.connect(self.start_exploring)
        self.stop = QtWidgets.QPushButton('Stop')
        self.stop.setObjectName('stop')
        self.stop.clicked.connect(lambda: node.arm(False, 'Stopped by user'))
        toolbar.addWidget(self.start)
        toolbar.addWidget(self.stop)
        for title, action in [('Pause Gazebo', lambda: node.set_sim_paused(True)),
                              ('Resume Gazebo', lambda: node.set_sim_paused(False)),
                              ('Save report + map', self.capture)]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(action)
            toolbar.addWidget(button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.banner = QtWidgets.QLabel('Connecting to Gazebo…')
        self.banner.setObjectName('banner')
        self.banner.setWordWrap(True)
        layout.addWidget(self.banner)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.views = QtWidgets.QTabWidget()
        mapping = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(mapping)
        self.map_image = MapView()
        map_layout.addWidget(self.map_image, 1)
        self.map_status = QtWidgets.QLabel('Waiting for live map data')
        self.map_status.setWordWrap(True)
        map_layout.addWidget(self.map_status)
        legend_row = QtWidgets.QHBoxLayout()
        legend = QtWidgets.QLabel('White: observed free space  ·  Hatched: unknown  ·  Blue-gray: obstacles\nGold: unexplored boundaries  ·  Green: planned route + target  ·  Arrow: robot heading')
        legend.setWordWrap(True)
        legend_row.addWidget(legend, 1)
        fit = QtWidgets.QPushButton('Fit map')
        fit.clicked.connect(self.map_image.fit_map)
        legend_row.addWidget(fit)
        map_layout.addLayout(legend_row)
        self.views.addTab(mapping, 'Live SLAM map')
        camera = QtWidgets.QWidget()
        camera_layout = QtWidgets.QHBoxLayout(camera)
        self.images = []
        for title in ['Robot camera / depth source frame', 'Relative depth · warm = near']:
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(title))
            label = QtWidgets.QLabel('Waiting for camera…')
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumSize(280, 240)
            label.setStyleSheet('background:#080f16;border:1px solid #30465b;')
            column.addWidget(label, 1)
            camera_layout.addLayout(column)
            self.images.append(label)
        self.views.addTab(camera, 'Camera + depth')
        object_page=QtWidgets.QWidget()
        object_layout=QtWidgets.QVBoxLayout(object_page)
        self.objects_caption=QtWidgets.QLabel('Loading live YOLO object detection…')
        self.objects_caption.setWordWrap(True)
        object_layout.addWidget(self.objects_caption)
        self.objects_image=QtWidgets.QLabel('Waiting for a live camera frame')
        self.objects_image.setAlignment(QtCore.Qt.AlignCenter)
        self.objects_image.setMinimumSize(460,330)
        self.objects_image.setStyleSheet('background:#080f16;border:1px solid #30465b;')
        object_layout.addWidget(self.objects_image,1)
        self.views.addTab(object_page,'Live objects · YOLO')
        if getattr(node,'detector',None) is not None:self.views.setCurrentIndex(2)
        splitter.addWidget(self.views)
        tables = QtWidgets.QTabWidget()
        self.status_table = make_table(['Parameter', 'Value'])
        self.status_table.setObjectName('statusTable')
        tables.addTab(self.status_table, 'Status')
        sensor_page = QtWidgets.QWidget()
        sensor_layout = QtWidgets.QVBoxLayout(sensor_page)
        sensor_layout.addWidget(QtWidgets.QLabel('Ultrasonic distances'))
        self.range_table = make_table(['Sensor', 'Distance (m)', 'Age (ms)', 'Status'])
        self.range_table.setObjectName('rangeTable')
        self.range_table.setMaximumHeight(200)
        sensor_layout.addWidget(self.range_table)
        self.attitude_table = make_table(['Measurement', 'Value', 'Unit'])
        sensor_layout.addWidget(QtWidgets.QLabel('IMU and compass'))
        sensor_layout.addWidget(self.attitude_table, 1)
        tables.addTab(sensor_page, 'Sensors')
        self.object_table=make_table(['Object','Confidence','Box XYXY (px)'])
        self.object_table.setObjectName('objectsTable')
        tables.addTab(self.object_table,'Objects')
        splitter.addWidget(tables)
        splitter.setSizes([850, 430])
        layout.addWidget(splitter, 1)
        self.footer = QtWidgets.QLabel('Map-guided goals and visit memory · Space: stop · A: start · Wheel: zoom · Drag: pan')
        layout.addWidget(self.footer)
        self.shortcuts = []
        for key, action in [('Space', lambda: node.arm(False, 'Stopped by keyboard')),
                            ('A', self.start_exploring), ('Esc', self.close)]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def start_exploring(self):
        self.node.set_mode('explore')
        self.node.arm(True)
        self.banner.setText('Start requested · waiting for valid sensors and a fresh SLAM route')

    def capture(self):
        if self.export_job is not None and not self.export_job.done():
            return
        screenshot = self.grab().toImage()
        map_image = self.map_image.grab().toImage()
        def export():
            self.save()
            self.args.output.mkdir(parents=True, exist_ok=True)
            screenshot.save(str(self.args.output/'dashboard.png'))
            map_image.save(str(self.args.output/'map_view.png'))
        self.export_job = self.exports.submit(export)
        self.footer.setText('Saving report and map in the background…')

    def display_camera(self, frame, prediction):
        source = prediction.packet if prediction else frame
        if source is None or (source.id == self.last_depth_id and prediction is not None):
            return
        self.last_depth_id = source.id
        images = [source.bgr]
        if prediction:
            images.append(cv2.applyColorMap(np.uint8(np.clip(prediction.view.proximity, 0, 1)*255), cv2.COLORMAP_TURBO))
        for label, image in zip(self.images, images):
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            q = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            label.setPixmap(QtGui.QPixmap.fromImage(q).scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def display_objects(self,frame):
        from vbmd_depth_backend.robot_detection import annotate_objects
        detector=getattr(self.node,'detector',None)
        if detector is None:
            self.objects_caption.setText('Object detection disabled')
            return
        snapshot=detector.snapshot();prediction=snapshot['prediction']
        fresh=prediction is not None and snapshot['age_s']<1.
        if fresh:
            self.objects_caption.setText(f'{snapshot["status"]} · frame #{prediction.packet.id} · '
                f'{len(prediction.boxes)} objects · inference {prediction.inference_ms:.1f} ms · '
                f'frame age {snapshot["age_s"]*1000:.0f} ms')
            fill_table(self.object_table,[(b.label,f'{b.confidence:.0%}',', '.join(f'{v:.0f}' for v in b.xyxy)) for b in prediction.boxes])
            identity=('yolo',prediction.packet.id)
            if identity==self.last_object_id:return
            image=annotate_objects(prediction)
        else:
            reason=snapshot['error'].strip().splitlines()[-1] if snapshot['error'] else 'Waiting for a fresh detection frame'
            self.objects_caption.setText(snapshot['status']+' · '+reason)
            fill_table(self.object_table,[])
            if frame is None:
                self.objects_image.clear()
                self.objects_image.setText('Waiting for a fresh camera frame')
                return
            identity=('camera',frame.id)
            if identity==self.last_object_id:return
            image=frame.bgr
        self.last_object_id=identity
        rgb=np.ascontiguousarray(image[:,:,::-1])
        q=QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
        self.objects_image.setPixmap(QtGui.QPixmap.fromImage(q).scaled(self.objects_image.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation))

    def refresh(self):
        node = self.node
        frame, prediction, ranges, decision, active = node.snapshot()
        now = time.monotonic()
        slam = node.slam.snapshot()
        attitude, _, _ = node.navigation_snapshot()
        connected = frame is not None and now-frame.received < .65 and now-node.clock_received < 1.
        requested = node.start_requested or node.arming or active
        self.start.setEnabled(not requested)
        self.start.setText('Exploring' if active else ('Start requested…' if requested else 'Start exploring'))
        self.banner.setText(f'{decision.state} · {decision.reason}')
        revision = (slam.get('revision'), slam.get('plan_stamp'), tuple(slam['pose']))
        if revision != self.last_map_revision:
            self.map_image.display_snapshot(slam)
            self.last_map_revision = revision
        fresh = slam['age_s'] < .45
        label = 'LIVE' if fresh else 'STALE'
        age = f'{slam["age_s"]*1000:.0f} ms' if np.isfinite(slam['age_s']) else 'no frames'
        self.map_status.setText(f'{label} · update #{slam.get("revision", 0)} · age {age}\n{slam["status"]}')
        goal = slam['goal']
        goal_text = 'Selecting frontier' if goal is None else f'#{slam.get("goal_id",0)} · ({goal[0]:+.2f}, {goal[1]:+.2f}) m'
        number = lambda x: f'{x*1000:.0f} ms' if x is not None and np.isfinite(x) else 'Waiting'
        rows = [
            ('Gazebo connection', 'Connected' if connected else 'Paused / waiting for live data'),
            ('Movement request', 'Armed' if active else ('Pending' if requested else 'Stopped')),
            ('Control state', decision.state), ('Current interest', slam['interest']),
            ('Tracking', 'Valid' if slam['ready'] else ('Brief dropout — bounded slow correction' if decision.state=='TRACKING_BRIDGE' else 'Recovering / initializing — position held')),
            ('Frontier planner', slam['planner_state']), ('Target position', goal_text),
            ('Estimated X', f'{slam["pose"][0]:+.3f} m'), ('Estimated Y', f'{slam["pose"][1]:+.3f} m'),
            ('Estimated yaw', f'{np.degrees(slam["pose"][2]):+.1f}°'),
            ('Distance to destination', '—' if goal is None else f'{np.linalg.norm(np.asarray(goal)-slam["pose"][:2]):.2f} m'),
            ('Destinations reached', slam.get('completed_goals',0)),
            ('Destination policy', 'Locked until confirmed arrival'),
            ('Observed / free area', f'{slam["observed_m2"]:.2f} / {slam["free_m2"]:.2f} m²'),
            ('Remembered places', slam['visited_places']), ('Visual inliers', slam['inliers']),
            ('Keyframes / loop closures', f'{slam["keyframes"]} / {slam["loop_closures"]}'),
            ('Camera frame / age', f'#{frame.id} / {number(now-frame.received)}' if frame else 'Waiting'),
            ('Depth prediction age', number(now-prediction.packet.received) if prediction else 'Model warming up'),
            ('Inference latency', f'{prediction.inference_ms:.1f} ms' if prediction else self.worker.status),
            ('Planner age', number(slam.get('plan_age_s'))),
            ('Trusted pose age', number(slam.get('trusted_pose_age_s'))),
            ('Vision processing latency', f'{slam.get("processing_ms",0.):.1f} ms'),
            ('Commanded forward / yaw', f'{decision.linear:+.3f} m/s / {decision.angular:+.3f} rad/s'),
            ('Simulation control', node.world_control_status or 'Running' if connected else node.world_control_status or 'Waiting'),
        ]
        fill_table(self.status_table, rows)
        sensor_rows = []
        for side in ['front', 'left', 'rear', 'right']:
            reading = ranges.get(side)
            if reading is None:
                sensor_rows.append((side.title(), '—', '—', 'Waiting'))
            else:
                value = 'No echo' if reading.value == np.inf else ('Too close' if reading.value == -np.inf else f'{reading.value:.3f}')
                sensor_rows.append((side.title(), value, f'{reading.age_wall*1000:.0f}', 'Valid' if reading.usable() else 'Stale / invalid'))
        fill_table(self.range_table, sensor_rows)
        attitude_rows = [(name.title(), f'{attitude[name+"_deg"]:+.2f}', 'degrees') for name in ['roll', 'pitch', 'yaw']]
        attitude_rows.append(('Compass heading', f'{attitude["heading_deg"]:.2f}', 'degrees'))
        for title, values, unit in [('Acceleration', attitude['acceleration_m_s2'], 'm/s²'),
                                    ('Angular velocity', attitude['angular_velocity_deg_s'], '°/s'),
                                    ('Magnetic field', attitude['magnetic_field_uT'], 'µT')]:
            attitude_rows.extend((f'{title} {axis}', f'{value:+.3f}', unit) for axis, value in zip('XYZ', values))
        fill_table(self.attitude_table, attitude_rows)
        self.display_camera(frame, prediction)
        self.display_objects(frame)
        if self.export_job is not None and self.export_job.done():
            error = self.export_job.exception()
            self.footer.setText('Export failed: '+str(error) if error else 'Saved report, map and dashboard to '+str(self.args.output))
            self.export_job = None
        if self.args.duration and now-self.opened >= self.args.duration:
            self.capture()
            self.close()

    def closeEvent(self, event):
        self.node.arm(False, 'Dashboard closed')
        self.timer.stop()
        event.accept()


def run_window(node, worker, args, save):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName('JERRY Robot Explorer')
    window = RobotWindow(node, worker, args, save)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    window.show()
    try:
        app.exec()
    finally:
        window.exports.shutdown(wait=True, cancel_futures=False)
