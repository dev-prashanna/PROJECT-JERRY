import socket 
import time 

ESP32_IP="192.168.4.1"
ESP32_PORT=8888

sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(command: str):
    try:
        sock.sendto(command.encode('utf-8'),(ESP32_IP,ESP32_PORT))
    except Exception as e:
        print(f"Error sending command: {e}")

def set_speed(speed: int):
    clamped_speed = max(0,min(speed,255))
    send_command(f"V{clamped_speed}")

if __name__ == "__main__":

    set_speed(200)
    time.sleep(0.1)