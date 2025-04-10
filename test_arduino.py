import serial
import time

# Connect to Arduino
try:
    arduino = serial.Serial('COM7', 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print("Connected to Arduino")
    
    # Read the Arduino's startup message
    if arduino.in_waiting:
        print("Arduino says:", arduino.readline().decode().strip())
    
    # Test sending ON/OFF commands
    for _ in range(5):  # Test 5 times
        print("Sending ON")
        arduino.write("on\n".encode())
        time.sleep(1)
        
        if arduino.in_waiting:
            print("Arduino response:", arduino.readline().decode().strip())
            
        print("Sending OFF")
        arduino.write("off\n".encode())
        time.sleep(1)
        
        if arduino.in_waiting:
            print("Arduino response:", arduino.readline().decode().strip())
            
except Exception as e:
    print(f"Error: {e}")
finally:
    try:
        arduino.close()
    except:
        pass