"""Small ROS lifecycle helpers; import only in ROS-facing processes."""
import time
import threading
from dataclasses import asdict
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from jerry_interfaces.srv import Authority


def authority_request(operation, binding=None, **kw):
    req = Authority.Request(operation=operation)
    if binding is not None:
        for name, value in asdict(binding).items():
            setattr(req, name, value)
    for name, value in kw.items():
        setattr(req, name, value)
    return req


def wall_timer(node, interval, callback):
    return node.create_timer(interval, callback, clock=Clock(clock_type=ClockType.STEADY_TIME))


def wait_future(future, timeout=3.):
    end = time.monotonic()+timeout
    while rclpy.ok() and not future.done():
        if time.monotonic() >= end:
            raise TimeoutError('ROS operation deadline exceeded; reconcile before retry')
        time.sleep(.01)
    if not future.done():
        raise RuntimeError('ROS shutting down')
    return future.result()


def run_node(factory):
    import signal
    from rclpy.signals import SignalHandlerOptions
    from rclpy.executors import ExternalShutdownException
    stop = threading.Event()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    node = factory()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=.1)
    except (KeyboardInterrupt,ExternalShutdownException):
        pass
    finally:
        if hasattr(node, 'close'):
            node.close()
        # Flush stop and terminal bookkeeping while services and ROS handles are still alive.
        end = time.monotonic()+.25
        while time.monotonic()<end and rclpy.ok():
            executor.spin_once(timeout_sec=.02)
        executor.shutdown(timeout_sec=3.)
        node.destroy_node()
        rclpy.try_shutdown()
