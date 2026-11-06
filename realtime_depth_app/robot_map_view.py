"""Antialiased, metric vector map presentation; occupancy data stays unchanged."""
import math
import cv2
import numpy as np
from scipy import ndimage
from PySide6 import QtCore, QtGui, QtWidgets

UNITS = 100.0  # scene units per metre


def scene_point(xy):
    return QtCore.QPointF(float(xy[0])*UNITS, -float(xy[1])*UNITS)


def occupancy_path(mask, origin, resolution):
    """Preserve holes, simplifying outlines by less than one source cell."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    path = QtGui.QPainterPath()
    path.setFillRule(QtCore.Qt.OddEvenFill)
    for contour in contours:
        contour = cv2.approxPolyDP(contour, .55, True).reshape(-1, 2)
        points = origin+(contour+.5)*resolution
        if len(points) < 3:
            if len(points)==1:
                center=scene_point(points[0]);width=resolution*UNITS
                path.addRect(QtCore.QRectF(center.x()-width/2,center.y()-width/2,width,width))
            elif len(points)==2:
                line=QtGui.QPainterPath(scene_point(points[0]));line.lineTo(scene_point(points[1]))
                stroke=QtGui.QPainterPathStroker();stroke.setWidth(resolution*UNITS)
                stroke.setCapStyle(QtCore.Qt.SquareCap)
                path.addPath(stroke.createStroke(line))
            continue
        path.moveTo(scene_point(points[0]))
        for point in points[1:]:
            path.lineTo(scene_point(point))
        path.closeSubpath()
    return path


def frontier_markers(snapshot):
    points = np.asarray(snapshot.get('frontiers', []), dtype=float)
    if points.size == 0:
        return []
    grid = snapshot['grid']
    origin = np.asarray(snapshot['origin'])
    resolution = snapshot['resolution']
    cells = np.floor((points-origin)/resolution).astype(int)
    valid = (cells >= 0).all(axis=1) & (cells[:, 0] < grid.shape[1]) & (cells[:, 1] < grid.shape[0])
    points, cells = points[valid], cells[valid]
    mask = np.zeros(grid.shape, bool)
    mask[cells[:, 1], cells[:, 0]] = True
    labels, _ = ndimage.label(ndimage.binary_dilation(mask))
    groups = {}
    for point, (x, y) in zip(points, cells):
        groups.setdefault(int(labels[y, x]), []).append(point)
    markers = []
    for values in sorted(groups.values(), key=len, reverse=True)[:12]:
        values = np.asarray(values)
        center = values.mean(axis=0)
        markers.append(values[np.argmin(np.linalg.norm(values-center, axis=1))])
    return markers


class VectorMapItem(QtWidgets.QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.bounds = QtCore.QRectF(-200, -200, 400, 400)
        self.free = QtGui.QPainterPath()
        self.occupied = QtGui.QPainterPath()
        self.grid = None
        self.snapshot = None
        self.markers = []
        self.plan_stamp = None
        self.epoch = None

    def boundingRect(self):
        return self.bounds

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot
        grid = snapshot['grid']
        origin = np.asarray(snapshot['origin'])
        resolution = snapshot['resolution']
        changed = self.grid is None or not np.array_equal(self.grid, grid)
        if changed:
            self.grid = grid.copy()
            self.free = occupancy_path(grid == 0, origin, resolution)
            self.occupied = occupancy_path(grid == 100, origin, resolution)
            ys, xs = np.where(grid >= 0)
            if len(xs):
                lo = origin+np.array([xs.min(), ys.min()])*resolution-.5
                hi = origin+np.array([xs.max()+1, ys.max()+1])*resolution+.5
                center = (lo+hi)/2
                extent = np.maximum(hi-lo, 3.)
                lo, hi = center-extent/2, center+extent/2
                bounds = QtCore.QRectF(float(lo[0])*UNITS, -float(hi[1])*UNITS,
                                      float(hi[0]-lo[0])*UNITS, float(hi[1]-lo[1])*UNITS)
                if self.epoch == snapshot.get('epoch'):
                    bounds = self.bounds.united(bounds)
                self.prepareGeometryChange()
                self.bounds = bounds
                self.epoch = snapshot.get('epoch')
        if changed or self.plan_stamp != snapshot.get('plan_stamp'):
            self.markers = frontier_markers(snapshot)
            self.plan_stamp = snapshot.get('plan_stamp')
        self.update()

    @staticmethod
    def pen(color, width, style=QtCore.Qt.SolidLine):
        pen = QtGui.QPen(QtGui.QColor(color), width, style)
        pen.setCosmetic(True)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        return pen

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.bounds, QtGui.QColor('#e1e7ed'))
        hatch = QtGui.QBrush(QtGui.QColor('#c9d3dc'), QtCore.Qt.BDiagPattern)
        painter.fillRect(self.bounds, hatch)
        painter.setPen(self.pen('#bac7d2', 1.))
        painter.setBrush(QtGui.QColor('#fbfcfd'))
        painter.drawPath(self.free)
        # Metric grid is independent of sensor cells; it communicates room scale.
        for coordinate in np.arange(math.floor(self.bounds.left()/50)*50, self.bounds.right()+1, 50):
            painter.setPen(self.pen('#c3ced8' if coordinate % 100 == 0 else '#dce3e9', .8))
            painter.drawLine(QtCore.QLineF(coordinate, self.bounds.top(), coordinate, self.bounds.bottom()))
        for coordinate in np.arange(math.floor(self.bounds.top()/50)*50, self.bounds.bottom()+1, 50):
            painter.setPen(self.pen('#c3ced8' if coordinate % 100 == 0 else '#dce3e9', .8))
            painter.drawLine(QtCore.QLineF(self.bounds.left(), coordinate, self.bounds.right(), coordinate))
        painter.setPen(self.pen('#26394b', 1.5))
        painter.setBrush(QtGui.QColor('#40586e'))
        painter.drawPath(self.occupied)
        if self.snapshot is None:
            return
        s = self.snapshot
        painter.setBrush(QtGui.QColor('#dfa638'))
        painter.setPen(self.pen('#a97115', 1.))
        for point in self.markers:
            painter.drawEllipse(scene_point(point), 3.5, 3.5)
        for points, color, width in [(s.get('trajectory', []), '#4d8ab6', 1.6), (s.get('path', []), '#12856f', 3.)]:
            if len(points) > 1:
                path = QtGui.QPainterPath(scene_point(points[0]))
                for point in points[1:]:
                    path.lineTo(scene_point(point))
                painter.setPen(self.pen(color, width))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawPath(path)
        goal = s.get('goal')
        if goal is not None:
            center = scene_point(goal)
            painter.setBrush(QtGui.QColor('#ffffff'))
            painter.setPen(self.pen('#12856f', 2.))
            painter.drawEllipse(center, 7, 7)
            painter.drawLine(center+QtCore.QPointF(-10, 0), center+QtCore.QPointF(10, 0))
            painter.drawLine(center+QtCore.QPointF(0, -10), center+QtCore.QPointF(0, 10))
            painter.setFont(QtGui.QFont('Sans Serif', 9))
            painter.setPen(QtGui.QColor('#086653'))
            painter.drawText(center+QtCore.QPointF(12, -10), 'Explore here')
        pose = s['pose']
        center = scene_point(pose)
        color = '#db4d47' if s.get('ready') else '#cf8b27'
        painter.setPen(self.pen('#ffffff', 2.))
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(center, 8.5, 8.5)
        direction = np.array([math.cos(pose[2]), math.sin(pose[2])])
        side = np.array([-direction[1], direction[0]])
        triangle = QtGui.QPolygonF([scene_point(np.asarray(pose[:2])+.30*direction),
                                   scene_point(np.asarray(pose[:2])+.09*direction+.07*side),
                                   scene_point(np.asarray(pose[:2])+.09*direction-.07*side)])
        painter.drawPolygon(triangle)
        painter.setPen(self.pen('#71879b', 1.))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(scene_point([0, 0]), 3, 3)
        painter.setFont(QtGui.QFont('Sans Serif', 8))
        painter.setPen(QtGui.QColor('#536b7e'))
        painter.drawText(scene_point([.07, -.12]), 'Start')


class MapView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.map_scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.map_scene)
        self.item = VectorMapItem()
        self.map_scene.addItem(self.item)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.auto_fit = True
        self.setMinimumSize(460, 330)
        self.setBackgroundBrush(QtGui.QColor('#e1e7ed'))

    def display_snapshot(self, snapshot):
        self.item.set_snapshot(snapshot)
        self.map_scene.setSceneRect(self.item.boundingRect())
        if self.auto_fit:
            self.fit_map()

    def fit_map(self):
        self.auto_fit = True
        self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_map()

    def wheelEvent(self, event):
        self.auto_fit = False
        self.scale(1.15 if event.angleDelta().y() > 0 else 1/1.15, 1.15 if event.angleDelta().y() > 0 else 1/1.15)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.auto_fit = False
        super().mousePressEvent(event)

    def drawForeground(self, painter, rect):
        # Fixed-pixel legend/scale remains legible when zooming.
        painter.save()
        painter.resetTransform()
        painter.setFont(QtGui.QFont('Sans Serif', 9))
        origin = self.mapFromScene(0, 0)
        metre = abs(self.mapFromScene(UNITS, 0).x()-origin.x())
        length, label = metre, '1 m'
        if length > self.viewport().width()/3:
            length, label = length/2, '0.5 m'
        if length > 5:
            x, y = 18, self.viewport().height()-24
            painter.setBrush(QtGui.QColor(255, 255, 255, 230))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(QtCore.QRectF(x-8, y-23, length+26, 42), 5, 5)
            painter.setPen(QtGui.QPen(QtGui.QColor('#28465c'), 2))
            painter.drawLine(QtCore.QLineF(x, y, x+length, y))
            painter.drawLine(x, y-4, x, y+4)
            painter.drawLine(QtCore.QLineF(x+length, y-4, x+length, y+4))
            painter.drawText(QtCore.QPointF(x+length/2-10, y-7), label)
        painter.setPen(QtGui.QColor('#365269'))
        painter.drawText(14, 20, 'Metric map  ·  grid 0.5 m  ·  sensor cells 5 cm')
        painter.restore()
