"""Asynchronous text/PTT console. Stop remains available during a model request."""
import sys
import threading
from uuid import uuid4
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from jerry_interfaces.srv import Ask


def main():
    rclpy.init()
    node = Node('jerry_console')
    ask = node.create_client(Ask,'/jerry/dialogue/ask')
    stop = node.create_client(Trigger,'/jerry/motion/stop')
    cancel = node.create_client(Trigger,'/jerry/tasks/cancel')
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin,daemon=True)
    thread.start()
    print('Jerry console: explore the room | status | what can you see? | what did you find earlier? | /listen | stop | /quit',flush=True)
    try:
        for line in sys.stdin:
            text = line.strip()
            if text in ('/quit','/exit'):break
            if not text:continue
            if text.lower() in ('stop','/stop'):
                stop.call_async(Trigger.Request())
                cancel.call_async(Trigger.Request())
                text = 'stop'
            if not ask.service_is_ready():
                print('Dialogue starting; direct Stop remains available.',flush=True)
                continue
            future = ask.call_async(Ask.Request(request_id=uuid4().hex,text=text))
            def done(f):
                result = f.result()
                print(('Jerry: ' if result.accepted else 'Unavailable: ')+result.text,flush=True)
            future.add_done_callback(done)
    except KeyboardInterrupt:
        pass
    finally:
        stop.call_async(Trigger.Request())
        cancel.call_async(Trigger.Request())
        executor.shutdown(timeout_sec=2.)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
