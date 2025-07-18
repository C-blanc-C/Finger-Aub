import serial
import json
import time
import threading
import numpy as np
from collections import deque
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class OptimizedMLPipeline:
    """Optimized ML pipeline for real-time performance"""
    
    def __init__(self, use_tensorrt=True):
        self.use_tensorrt = use_tensorrt
        
        if use_tensorrt:
            self._initialize_tensorrt()
        else:
            self._initialize_onnx()
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        
    def _initialize_tensorrt(self):
        """Initialize TensorRT for optimized inference"""
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load TensorRT engine
        with open("yolov8_engine.trt", "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
    def _initialize_onnx(self):
        """Initialize ONNX Runtime as fallback"""
        # Configure ONNX Runtime for GPU
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # YOLOv8 model
        self.yolo_session = ort.InferenceSession("yolov8m.onnx", providers=providers)
        
        # Depth model
        self.depth_session = ort.InferenceSession("midas_v2.onnx", providers=providers)
        
    def _allocate_buffers(self):
        """Allocate CUDA buffers for TensorRT"""
        self.buffers = []
        self.outputs = {}
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.buffers.append({'host': host_mem, 'device': device_mem})
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.input_buffer = {'host': host_mem, 'device': device_mem}
            else:
                self.outputs[binding] = {'host': host_mem, 'device': device_mem}
    
    def optimize_model_for_deployment(self, onnx_path, output_path):
        """Convert ONNX model to TensorRT for deployment"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with trt.Builder(TRT_LOGGER) as builder:
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision
            
            # Parse ONNX
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            with open(onnx_path, 'rb') as model:
                parser.parse(model.read())
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            # Serialize
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {output_path}")


class HardwareInterface:
    """Robust hardware interface with error handling and monitoring"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
        
        # Thread-safe command queue
        self.command_queue = deque(maxlen=10)
        self.feedback_data = {}
        
        # Communication thread
        self.comm_thread = None
        self.running = False
        
        # Performance metrics
        self.latency_buffer = deque(maxlen=100)
        self.error_count = 0
        
        # Connect to hardware
        self.connect()
        
    def connect(self):
        """Establish connection with Arduino"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.1
            )
            time.sleep(2)  # Wait for Arduino reset
            
            # Clear any startup messages
            self.serial_conn.reset_input_buffer()
            
            self.connected = True
            self.running = True
            
            # Start communication thread
            self.comm_thread = threading.Thread(target=self._communication_loop)
            self.comm_thread.daemon = True
            self.comm_thread.start()
            
            print(f"Connected to Arduino on {self.port}")
            
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            self.connected = False
    
    def _communication_loop(self):
        """Main communication loop"""
        while self.running:
            try:
                # Send queued commands
                if self.command_queue and self.serial_conn:
                    command = self.command_queue.popleft()
                    self._send_command(command)
                
                # Read feedback
                if self.serial_conn and self.serial_conn.in_waiting:
                    self._read_feedback()
                
                time.sleep(0.001)  # 1ms loop time
                
            except Exception as e:
                self.error_count += 1
                print(f"Communication error: {e}")
                
                if self.error_count > 10:
                    self.reconnect()
    
    def _send_command(self, command):
        """Send command with timing"""
        start_time = time.time()
        
        try:
            self.serial_conn.write(command.encode())
            self.serial_conn.flush()
            
            # Track latency
            latency = (time.time() - start_time) * 1000
            self.latency_buffer.append(latency)
            
        except serial.SerialException as e:
            print(f"Send error: {e}")
            self.error_count += 1
    
    def _read_feedback(self):
        """Read and parse feedback data"""
        try:
            line = self.serial_conn.readline().decode().strip()
            
            if line.startswith('{'):
                # JSON feedback data
                self.feedback_data = json.loads(line)
                self.feedback_data['timestamp'] = time.time()
                
            elif line.startswith('WARNING'):
                print(f"Hardware warning: {line}")
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            pass  # Ignore malformed data
    
    def send_actuator_commands(self, commands):
        """Queue actuator commands"""
        # Format: T:XX,I:XX,M:XX,R:XX,P:XX\n
        cmd_str = f"T:{commands.get('thumb', 0):.0f},"
        cmd_str += f"I:{commands.get('index', 0):.0f},"
        cmd_str += f"M:{commands.get('middle', 0):.0f},"
        cmd_str += f"R:{commands.get('ring', 0):.0f},"
        cmd_str += f"P:{commands.get('pinky', 0):.0f}\n"
        
        self.command_queue.append(cmd_str)
    
    def get_feedback(self):
        """Get latest feedback data"""
        return self.feedback_data.copy()
    
    def get_performance_metrics(self):
        """Get system performance metrics"""
        if self.latency_buffer:
            avg_latency = np.mean(self.latency_buffer)
            max_latency = np.max(self.latency_buffer)
            min_latency = np.min(self.latency_buffer)
        else:
            avg_latency = max_latency = min_latency = 0
        
        return {
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'error_count': self.error_count,
            'connected': self.connected
        }
    
    def reconnect(self):
        """Attempt to reconnect to hardware"""
        print("Attempting to reconnect...")
        self.disconnect()
        time.sleep(1)
        self.connect()
        self.error_count = 0
    
    def disconnect(self):
        """Clean disconnect"""
        self.running = False
        
        if self.comm_thread:
            self.comm_thread.join(timeout=1)
        
        if self.serial_conn:
            self.serial_conn.close()
        
        self.connected = False
        print("Disconnected from hardware")
    
    def emergency_stop(self):
        """Send emergency stop command"""
        if self.connected:
            self.serial_conn.write(b"STOP\n")
            self.serial_conn.flush()
            print("Emergency stop sent!")
    
    def calibrate(self):
        """Initiate calibration sequence"""
        if self.connected:
            self.serial_conn.write(b"CALIB\n")
            print("Calibration started...")


class SystemOptimizer:
    """System-wide optimization strategies"""
    
    @staticmethod
    def optimize_camera_settings(cap):
        """Optimize camera for low latency"""
        # Disable auto-exposure for consistent timing
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        
        # Set optimal resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Maximum FPS
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Reduce buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Use MJPEG for faster decode
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    @staticmethod
    def setup_process_priority():
        """Set high process priority (Linux)"""
        try:
            import os
            os.nice(-10)  # Increase priority
        except:
            pass
    
    @staticmethod
    def profile_performance(func):
        """Decorator for performance profiling"""
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            print(f"{func.__name__} took {(end - start) * 1000:.2f}ms")
            return result
        return wrapper


class AdaptiveController:
    """Adaptive control with learning capabilities"""
    
    def __init__(self):
        self.grip_history = deque(maxlen=1000)
        self.success_patterns = {}
        
    def learn_from_feedback(self, object_type, grip_pattern, force_feedback, success):
        """Learn successful grip patterns"""
        entry = {
            'object': object_type,
            'grip': grip_pattern,
            'forces': force_feedback,
            'success': success,
            'timestamp': time.time()
        }
        
        self.grip_history.append(entry)
        
        # Update success patterns
        if success:
            if object_type not in self.success_patterns:
                self.success_patterns[object_type] = []
            
            self.success_patterns[object_type].append({
                'grip': grip_pattern,
                'avg_force': np.mean(list(force_feedback.values()))
            })
    
    def get_optimal_grip(self, object_type):
        """Get optimal grip based on learned patterns"""
        if object_type in self.success_patterns:
            patterns = self.success_patterns[object_type]
            
            # Find pattern with best force distribution
            best_pattern = min(patterns, key=lambda x: x['avg_force'])
            return best_pattern['grip']
        
        return None
    
    def adjust_grip_for_force(self, current_grip, forces, target_force=5.0):
        """Dynamically adjust grip based on force feedback"""
        adjusted_grip = current_grip.copy()
        
        for finger, force in forces.items():
            if force < target_force * 0.8:
                # Increase grip
                adjusted_grip[finger] = min(current_grip[finger] + 5, 100)
            elif force > target_force * 1.2:
                # Decrease grip
                adjusted_grip[finger] = max(current_grip[finger] - 5, 0)
        
        return adjusted_grip


# Complete integrated system
class RoboticHandSystem:
    """Complete robotic hand control system"""
    
    def __init__(self):
        print("Initializing Robotic Hand System...")
        
        # Initialize components
        self.ml_pipeline = OptimizedMLPipeline(use_tensorrt=True)
        self.hardware = HardwareInterface()
        self.controller = AdaptiveController()
        
        # System state
        self.running = False
        self.current_grip = {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
        
        # Performance monitoring
        self.system_metrics = {
            'total_cycles': 0,
            'avg_cycle_time': 0,
            'max_cycle_time': 0,
            'success_rate': 0
        }
        
    def run(self):
        """Main control loop"""
        self.running = True
        
        # Set process priority
        SystemOptimizer.setup_process_priority()
        
        # Initialize camera with optimizations
        cap = cv2.VideoCapture(0)
        SystemOptimizer.optimize_camera_settings(cap)
        
        print("System running. Press 'q' to quit.")
        
        cycle_times = deque(maxlen=100)
        
        try:
            while self.running:
                cycle_start = time.perf_counter()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # ML inference
                commands, inference_time = self.ml_pipeline.process_frame(frame)
                
                # Get hardware feedback
                feedback = self.hardware.get_feedback()
                forces = feedback.get('force', {})
                
                # Adaptive control
                if forces:
                    commands = self.controller.adjust_grip_for_force(
                        commands, 
                        dict(zip(['thumb', 'index', 'middle', 'ring', 'pinky'], forces))
                    )
                
                # Send commands
                self.hardware.send_actuator_commands(commands)
                self.current_grip = commands
                
                # Update metrics
                cycle_time = (time.perf_counter() - cycle_start) * 1000
                cycle_times.append(cycle_time)
                self.system_metrics['total_cycles'] += 1
                
                # Display
                self._display_status(frame, inference_time, cycle_time)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hardware.disconnect()
            
            # Print final metrics
            self._print_metrics(cycle_times)
    
    def _display_status(self, frame, inference_time, cycle_time):
        """Display system status on frame"""
        # Add status overlay
        status_text = [
            f"Inference: {inference_time:.1f}ms",
            f"Cycle: {cycle_time:.1f}ms",
            f"FPS: {1000/cycle_time:.1f}"
        ]
        
        y_offset = 30
        for text in status_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Display grip positions
        grip_text = "Grip: " + ", ".join([f"{v:.0f}" for v in self.current_grip.values()])
        cv2.putText(frame, grip_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Robotic Hand Control", frame)
    
    def _print_metrics(self, cycle_times):
        """Print system performance metrics"""
        print("\n=== System Performance Metrics ===")
        print(f"Total cycles: {self.system_metrics['total_cycles']}")
        print(f"Average cycle time: {np.mean(cycle_times):.2f}ms")
        print(f"Max cycle time: {np.max(cycle_times):.2f}ms")
        print(f"Min cycle time: {np.min(cycle_times):.2f}ms")
        
        hw_metrics = self.hardware.get_performance_metrics()
        print(f"\nHardware latency: {hw_metrics['avg_latency_ms']:.2f}ms")
        print(f"Communication errors: {hw_metrics['error_count']}")


if __name__ == "__main__":
    # Run the complete system
    system = RoboticHandSystem()
    system.run()