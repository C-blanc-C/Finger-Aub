import pybullet as p
import pybullet_data
import numpy as np
import time
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class FingerJoint:
    """Represents a single finger joint"""
    joint_id: int
    min_angle: float
    max_angle: float
    name: str
    
class RoboticHandSimulator:
    """PyBullet-based robotic hand simulator"""
    
    def __init__(self, use_gui=True, hand_urdf_path=None):
        # Initialize PyBullet
        if use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robotic hand
        self.hand_id = self._load_hand_model(hand_urdf_path)
        
        # Initialize joint mapping
        self._initialize_joints()
        
        # Command queue for thread-safe updates
        self.command_queue = queue.Queue()
        
        # Start simulation thread
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop)
        self.sim_thread.start()
    
    def _load_hand_model(self, urdf_path):
        """Load hand URDF model or create simple representation"""
        if urdf_path:
            return p.loadURDF(urdf_path, [0, 0, 0.2])
        else:
            # Create simplified hand model
            return self._create_simple_hand()
    
    def _create_simple_hand(self):
        """Create a simplified 5-finger hand model"""
        # Base/palm
        palm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.08, 0.01],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        palm_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.08, 0.01]
        )
        
        mass = 0.5
        base_position = [0, 0, 0.2]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create multi-body with fingers
        link_masses = []
        link_collision_shapes = []
        link_visual_shapes = []
        link_positions = []
        link_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        
        # Finger parameters
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_offsets = [
            [-0.04, -0.06, 0.01],  # thumb
            [-0.02, 0.08, 0.01],   # index
            [0, 0.08, 0.01],       # middle
            [0.02, 0.08, 0.01],    # ring
            [0.04, 0.06, 0.01]     # pinky
        ]
        
        link_id = 0
        for finger_idx, (finger, offset) in enumerate(zip(fingers, finger_offsets)):
            # Each finger has 3 segments
            for segment in range(3):
                # Create finger segment
                segment_length = 0.03 if segment < 2 else 0.02
                segment_visual = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=0.007,
                    length=segment_length,
                    rgbaColor=[0.9, 0.7, 0.5, 1]
                )
                segment_collision = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=0.007,
                    height=segment_length
                )
                
                link_masses.append(0.01)
                link_collision_shapes.append(segment_collision)
                link_visual_shapes.append(segment_visual)
                
                if segment == 0:
                    # First segment connects to palm
                    link_positions.append(offset)
                    link_orientations.append([0, 0, 0, 1])
                    link_parent_indices.append(0)  # Palm is parent
                else:
                    # Subsequent segments connect to previous
                    link_positions.append([0, 0, segment_length])
                    link_orientations.append([0, 0, 0, 1])
                    link_parent_indices.append(link_id)
                
                link_joint_types.append(p.JOINT_REVOLUTE)
                link_joint_axes.append([1, 0, 0])  # Rotate around X axis
                
                link_id += 1
        
        # Create the multi-body
        hand_id = p.createMultiBody(
            baseMass=mass,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=palm_collision,
            baseVisualShapeIndex=palm_visual,
            basePosition=base_position,
            baseOrientation=base_orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, 0]] * len(link_masses),
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(link_masses),
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
        
        return hand_id
    
    def _initialize_joints(self):
        """Initialize joint mapping and constraints"""
        self.joints = {}
        self.finger_joints = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': [],
            'pinky': []
        }
        
        num_joints = p.getNumJoints(self.hand_id)
        
        # Map joints to fingers (assuming 3 joints per finger)
        joint_idx = 0
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            for segment in range(3):
                if joint_idx < num_joints:
                    joint_info = p.getJointInfo(self.hand_id, joint_idx)
                    joint = FingerJoint(
                        joint_id=joint_idx,
                        min_angle=0,
                        max_angle=np.pi/2,  # 90 degrees max bend
                        name=f"{finger}_segment_{segment}"
                    )
                    self.finger_joints[finger].append(joint)
                    self.joints[joint.name] = joint
                    
                    # Set joint limits
                    p.changeDynamics(
                        self.hand_id, 
                        joint_idx,
                        jointLowerLimit=joint.min_angle,
                        jointUpperLimit=joint.max_angle
                    )
                    
                    joint_idx += 1
    
    def set_actuator_positions(self, actuator_commands: Dict[str, float]):
        """Set finger positions based on actuator commands (0-100%)"""
        self.command_queue.put(actuator_commands)
    
    def _actuator_to_joint_angles(self, actuator_percent: float) -> List[float]:
        """Convert actuator percentage to joint angles"""
        # Map 0-100% to 0-90 degrees for each joint
        # Distribute actuation across all joints in finger
        max_angle = np.pi / 2  # 90 degrees
        
        # Non-linear mapping for more natural motion
        # First joint bends less than distal joints
        joint_ratios = [0.6, 0.8, 1.0]  # Proximal to distal
        
        angles = []
        for ratio in joint_ratios:
            angle = (actuator_percent / 100.0) * max_angle * ratio
            angles.append(angle)
        
        return angles
    
    def _simulation_loop(self):
        """Main simulation loop"""
        current_positions = {
            'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0
        }
        
        while self.running:
            # Process command queue
            while not self.command_queue.empty():
                try:
                    new_commands = self.command_queue.get_nowait()
                    for finger, percent in new_commands.items():
                        if finger in current_positions:
                            current_positions[finger] = percent
                except queue.Empty:
                    break
            
            # Update joint positions
            for finger, percent in current_positions.items():
                if finger in self.finger_joints:
                    angles = self._actuator_to_joint_angles(percent)
                    for joint, angle in zip(self.finger_joints[finger], angles):
                        p.setJointMotorControl2(
                            self.hand_id,
                            joint.joint_id,
                            p.POSITION_CONTROL,
                            targetPosition=angle,
                            force=1.0
                        )
            
            # Step simulation
            p.stepSimulation()
            time.sleep(1/240)
    
    def add_object(self, object_type='sphere', position=[0, 0.15, 0.1], size=0.03):
        """Add an object to grasp"""
        if object_type == 'sphere':
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=size,
                rgbaColor=[1, 0, 0, 1]
            )
        elif object_type == 'cube':
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size]*3)
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[size]*3,
                rgbaColor=[0, 1, 0, 1]
            )
        elif object_type == 'cylinder':
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=size, height=size*3)
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=size, 
                length=size*3,
                rgbaColor=[0, 0, 1, 1]
            )
        
        object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        return object_id
    
    def get_contact_forces(self):
        """Get contact forces for each finger"""
        contact_forces = {
            'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0
        }
        
        # Get all contact points
        contact_points = p.getContactPoints(self.hand_id)
        
        for contact in contact_points:
            # contact[3] is the link index on bodyA (hand)
            link_index = contact[3]
            
            # Find which finger this link belongs to
            for finger, joints in self.finger_joints.items():
                for joint in joints:
                    if joint.joint_id == link_index:
                        # contact[9] is the normal force
                        contact_forces[finger] += abs(contact[9])
                        break
        
        return contact_forces
    
    def close(self):
        """Clean up simulation"""
        self.running = False
        self.sim_thread.join()
        p.disconnect()


class SimulationVisualizer:
    """Real-time visualization of simulation data"""
    
    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Actuator positions plot
        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.positions = {f: [] for f in self.fingers}
        self.forces = {f: [] for f in self.fingers}
        self.time_data = []
        
        # Set up plots
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Actuator Position (%)')
        self.ax1.set_title('Finger Actuator Positions')
        self.ax1.set_ylim(0, 100)
        
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Contact Force (N)')
        self.ax2.set_title('Finger Contact Forces')
        
        # Initialize lines
        self.position_lines = {}
        self.force_lines = {}
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for finger, color in zip(self.fingers, colors):
            line, = self.ax1.plot([], [], label=finger, color=color)
            self.position_lines[finger] = line
            
            line, = self.ax2.plot([], [], label=finger, color=color)
            self.force_lines[finger] = line
        
        self.ax1.legend()
        self.ax2.legend()
        
        # Animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        
    def update_data(self, time_point, positions, forces):
        """Update visualization data"""
        self.time_data.append(time_point)
        
        # Keep only last 10 seconds of data
        if len(self.time_data) > 200:
            self.time_data.pop(0)
            for finger in self.fingers:
                self.positions[finger].pop(0)
                self.forces[finger].pop(0)
        
        for finger in self.fingers:
            self.positions[finger].append(positions.get(finger, 0))
            self.forces[finger].append(forces.get(finger, 0))
    
    def update_plot(self, frame):
        """Update plot animation"""
        lines = []
        
        if self.time_data:
            # Update position lines
            for finger in self.fingers:
                self.position_lines[finger].set_data(self.time_data, self.positions[finger])
                lines.append(self.position_lines[finger])
            
            # Update force lines
            for finger in self.fingers:
                self.force_lines[finger].set_data(self.time_data, self.forces[finger])
                lines.append(self.force_lines[finger])
            
            # Adjust x-axis limits
            if len(self.time_data) > 1:
                self.ax1.set_xlim(self.time_data[0], self.time_data[-1])
                self.ax2.set_xlim(self.time_data[0], self.time_data[-1])
                
                # Adjust y-axis for forces
                max_force = max([max(self.forces[f]) if self.forces[f] else 0 
                               for f in self.fingers] + [1])
                self.ax2.set_ylim(0, max_force * 1.1)
        
        return lines


# Integration example
def test_simulation_with_ml_pipeline():
    """Test the simulation with ML pipeline integration"""
    
    # Initialize components
    ml_pipeline = RoboticHandMLPipeline()
    simulator = RoboticHandSimulator(use_gui=True)
    actuator_controller = ActuatorController()
    
    # Add test object
    simulator.add_object('cylinder', position=[0, 0.12, 0.05])
    
    # Main control loop
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = ml_pipeline.cap.read()
            if not ret:
                break
            
            # Process with ML pipeline
            commands, inference_time = ml_pipeline.process_frame(frame)
            
            # Send to simulation
            simulator.set_actuator_positions(commands)
            
            # Get feedback
            contact_forces = simulator.get_contact_forces()
            
            # Display results
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        ml_pipeline.cap.release()
        cv2.destroyAllWindows()
        simulator.close()


if __name__ == "__main__":
    # Run simulation test
    test_simulation_with_ml_pipeline()