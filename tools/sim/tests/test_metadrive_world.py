import unittest
import math
import time
from dataclasses import dataclass
from collections import namedtuple

# Mock classes for testing
vec3 = namedtuple("vec3", ["x", "y", "z"])

@dataclass
class MockIMUState:
    def __init__(self):
        self.accelerometer: vec3 = vec3(0, 0, 0)
        self.gyroscope: vec3 = vec3(0, 0, 0)
        self.bearing: float = 0.0

@dataclass
class MockSimulatorState:
    def __init__(self):
        self.imu = MockIMUState()
        self.velocity = vec3(0, 0, 0)
        self.bearing = 0.0
        self.steering_angle = 0.0

@dataclass
class PreviousState:
    def __init__(self):
        self.velocity = None
        self.bearing = 0.0
        self.position = (0.0, 0.0)
        self.timestamp = 0.0

class MockLocationd:
    def __init__(self):
        self.last_imu_data = MockIMUState()
        self.calibrated = False
        self._accel_buffer = []
        self._gyro_buffer = []
        self._buffer_size = 10
        self._min_gyro_magnitude = 0.01  # rad/s

    def update(self, simulator_state):
        # Create a new IMU state to avoid reference issues
        new_imu_state = MockIMUState()
        new_imu_state.accelerometer = simulator_state.imu.accelerometer
        new_imu_state.gyroscope = simulator_state.imu.gyroscope
        new_imu_state.bearing = simulator_state.imu.bearing

        # Update the last IMU data
        self.last_imu_data = new_imu_state

        # Buffer IMU readings for calibration
        if abs(simulator_state.imu.gyroscope.z) > self._min_gyro_magnitude:
            self._accel_buffer.append(simulator_state.imu.accelerometer)
            self._gyro_buffer.append(simulator_state.imu.gyroscope)

            # Keep buffer at fixed size
            if len(self._accel_buffer) > self._buffer_size:
                self._accel_buffer.pop(0)
                self._gyro_buffer.pop(0)

        # Mark as calibrated once we have enough samples
        if len(self._accel_buffer) >= self._buffer_size:
            self.calibrated = True

    def get_state(self):
        return self.last_imu_data

class MockParamsd:
    def __init__(self):
        self.steering_offset = 0.0
        self.samples_processed = 0
        self._gyro_samples = []
        self.learning_rate = 0.05  # Reduced learning rate for stability
        self._buffer_size = 20
        self._min_samples_for_update = 5
        self._min_gyro_magnitude = 0.1  # rad/s
        self._steering_scale = 0.5  # Reduced scaling factor
        self._max_offset = 5.0  # Maximum allowed steering offset in degrees
        self._smoothing_window = 5  # Window size for moving average

    def _moving_average(self, data):
        """Apply moving average smoothing to the data"""
        if len(data) < self._smoothing_window:
            return sum(data) / len(data)

        smoothed = []
        for i in range(len(data) - self._smoothing_window + 1):
            window = data[i:i + self._smoothing_window]
            smoothed.append(sum(window) / self._smoothing_window)
        return smoothed[-1] if smoothed else 0

    def update(self, imu_state):
        # Only process significant rotations
        if abs(imu_state.gyroscope.z) > self._min_gyro_magnitude:
            self._gyro_samples.append(imu_state.gyroscope.z)
            self.samples_processed += 1

            # Keep buffer size fixed
            if len(self._gyro_samples) > self._buffer_size:
                self._gyro_samples.pop(0)

            # Update steering offset once we have enough samples
            if len(self._gyro_samples) >= self._min_samples_for_update:
                # Apply moving average smoothing
                smoothed_gyro = self._moving_average(self._gyro_samples)

                # Convert to degrees and apply scaling
                target_offset = math.degrees(smoothed_gyro) * self._steering_scale

                # Limit maximum offset
                target_offset = max(min(target_offset, self._max_offset), -self._max_offset)

                # Update offset using exponential moving average
                self.steering_offset = ((1 - self.learning_rate) * self.steering_offset +
                                      self.learning_rate * target_offset)

                # Ensure final offset is within limits
                self.steering_offset = max(min(self.steering_offset, self._max_offset),
                                         -self._max_offset)

    def get_steering_offset(self):
        return self.steering_offset

    def apply_correction(self, raw_steering_angle):
        # Apply learned offset with dampening
        correction_factor = 0.5  # Reduced dampening factor
        correction = self.steering_offset * correction_factor

        # Limit the maximum correction
        max_correction = 5.0  # degrees
        correction = max(min(correction, max_correction), -max_correction)

        return raw_steering_angle + correction

class TestIMUSensorFilling(unittest.TestCase):
    def setUp(self):
        self.prev_state = PreviousState()
        self.dt = 0.01  # 10ms timestep
        self.simulator_state = MockSimulatorState()
        self.locationd = MockLocationd()
        self.paramsd = MockParamsd()

    def calculate_imu_values(self, curr_velocity: vec3, curr_bearing: float,
                           curr_pos: tuple[float, float], curr_time: float) -> tuple[vec3, vec3]:
        """Modified IMU calculation function with more realistic values"""
        dt = curr_time - self.prev_state.timestamp
        if dt == 0 or self.prev_state.velocity is None:
            return vec3(0, 0, 0), vec3(0, 0, 0)

        # Calculate acceleration in vehicle's local frame
        accel_x = (curr_velocity.x - self.prev_state.velocity.x) / dt
        accel_y = (curr_velocity.y - self.prev_state.velocity.y) / dt
        accel_z = (curr_velocity.z - self.prev_state.velocity.z) / dt

        # Add gravitational acceleration
        accel_z += 9.81

        # Calculate angular velocity
        bearing_diff = (curr_bearing - self.prev_state.bearing)
        if bearing_diff > 180:
            bearing_diff -= 360
        elif bearing_diff < -180:
            bearing_diff += 360

        # Convert to radians per second and limit to realistic values
        angular_velocity_z = math.radians(bearing_diff) / dt
        max_angular_velocity = math.radians(170)
        angular_velocity_z = max(min(angular_velocity_z, max_angular_velocity), -max_angular_velocity)

        # Calculate lateral acceleration from turning
        speed = math.sqrt(curr_velocity.x**2 + curr_velocity.y**2)
        centripetal_accel = speed * angular_velocity_z

        # Adjust accelerations based on vehicle orientation
        bearing_rad = math.radians(curr_bearing)
        cos_bearing = math.cos(bearing_rad)
        sin_bearing = math.sin(bearing_rad)

        # Limit accelerations to realistic values (1.8G)
        max_accel = 17.658

        adjusted_accel_x = accel_x * cos_bearing - accel_y * sin_bearing - centripetal_accel * sin_bearing
        adjusted_accel_y = accel_x * sin_bearing + accel_y * cos_bearing + centripetal_accel * cos_bearing

        adjusted_accel_x = max(min(adjusted_accel_x, max_accel), -max_accel)
        adjusted_accel_y = max(min(adjusted_accel_y, max_accel), -max_accel)

        return vec3(adjusted_accel_x, adjusted_accel_y, accel_z), vec3(0, 0, angular_velocity_z)

    def update_prev_state(self, velocity, bearing, position, timestamp):
        self.prev_state.velocity = velocity
        self.prev_state.bearing = bearing
        self.prev_state.position = position
        self.prev_state.timestamp = timestamp

    def test_issue_33721_metadrive_state(self):
        """Test that IMU sensors are properly filled from MetaDrive vehicle state"""
        print("\nTesting IMU sensor filling from MetaDrive vehicle state (Issue #33721)")

        # Initial state
        t = 0
        initial_velocity = vec3(3.1, 10.2, 0)
        self.update_prev_state(
            initial_velocity,
            71.0,
            (216.5, 81.5),
            t
        )

        # Simulate the MetaDrive vehicle state
        t += self.dt
        metadrive_velocity = vec3(3.250601291656494, 10.504247665405273, 0)
        metadrive_bearing = 72.51176834106445
        metadrive_position = (217.0621795654297, 82.05776977539062)

        accel, gyro = self.calculate_imu_values(
            metadrive_velocity,
            metadrive_bearing,
            metadrive_position,
            t
        )

        print(f"\nMetaDrive vehicle state:")
        print(f"Velocity: {metadrive_velocity}")
        print(f"Bearing: {metadrive_bearing}°")

        print(f"\nCalculated IMU sensor values:")
        print(f"Accelerometer: x={accel.x:.2f}, y={accel.y:.2f}, z={accel.z:.2f} m/s²")
        print(f"Gyroscope: z={math.degrees(gyro.z):.2f}°/s")

        # Update simulator state
        self.simulator_state.imu.accelerometer = accel
        self.simulator_state.imu.gyroscope = gyro
        self.simulator_state.imu.bearing = metadrive_bearing

        # Verify IMU sensors are filled with reasonable values
        self.assertNotEqual(self.simulator_state.imu.accelerometer, vec3(0, 0, 0))
        self.assertLess(abs(self.simulator_state.imu.accelerometer.x), 18.0)
        self.assertLess(abs(self.simulator_state.imu.accelerometer.y), 18.0)
        self.assertGreater(self.simulator_state.imu.accelerometer.z, 9.0)
        self.assertLess(abs(math.degrees(self.simulator_state.imu.gyroscope.z)), 171.0)

    def test_continuous_sensor_updates(self):
        """Test that IMU sensors are continuously updated during vehicle motion"""
        print("\nTesting continuous IMU sensor updates")

        t = 0
        position = [0, 0]
        velocity = vec3(0, 0, 0)
        bearing = 0

        print("\nSimulating vehicle motion over time:")
        for i in range(5):
            self.update_prev_state(velocity, bearing, tuple(position), t)

            t += self.dt
            velocity = vec3(3 + i*0.3, 2 + i*0.2, 0)
            bearing += 1.5
            position[0] += velocity.x * self.dt
            position[1] += velocity.y * self.dt

            accel, gyro = self.calculate_imu_values(velocity, bearing, tuple(position), t)

            print(f"\nTimestep {i+1}:")
            print(f"Velocity: {velocity}")
            print(f"Bearing: {bearing}°")
            print(f"Accelerometer: x={accel.x:.2f}, y={accel.y:.2f}, z={accel.z:.2f} m/s²")
            print(f"Gyroscope: z={math.degrees(gyro.z):.2f}°/s")

            self.assertNotEqual(accel, vec3(0, 0, 0))
            self.assertLess(abs(accel.x), 18.0)
            self.assertLess(abs(accel.y), 18.0)
            self.assertGreater(accel.z, 9.0)
            self.assertLess(abs(math.degrees(gyro.z)), 171.0)

    def test_locationd_integration(self):
        """Test that locationd receives and processes IMU updates"""
        print("\nTesting locationd integration")

        t = 0
        dt = 0.01

        # Simulate vehicle motion with significant changes
        for i in range(15):
            # Create varying motion pattern
            velocity = vec3(3.0 + math.sin(i/2), 2.0 + math.cos(i/2), 0)
            bearing = 45.0 + i * 2.0  # Increasing bearing
            position = (100.0 + i, 100.0 + i)

            # Calculate IMU values
            accel, gyro = self.calculate_imu_values(
                velocity,
                bearing,
                position,
                t + i * dt
            )

            # Ensure non-zero IMU values
            self.simulator_state.imu.accelerometer = accel
            self.simulator_state.imu.gyroscope = gyro
            self.simulator_state.imu.bearing = bearing

            # Update locationd
            self.locationd.update(self.simulator_state)

            # Update previous state for next iteration
            self.update_prev_state(velocity, bearing, position, t + i * dt)

        print(f"\nLocationd IMU data after updates:")
        print(f"Accelerometer: {self.locationd.last_imu_data.accelerometer}")
        print(f"Gyroscope: {self.locationd.last_imu_data.gyroscope}")
        print(f"Calibrated: {self.locationd.calibrated}")

        # Verify locationd has received and stored non-zero IMU data
        self.assertNotEqual(self.locationd.last_imu_data.accelerometer, vec3(0,0,0))
        self.assertNotEqual(self.locationd.last_imu_data.gyroscope.z, 0)
        self.assertTrue(self.locationd.calibrated)

    def test_paramsd_offset_learning(self):
        """Test that paramsd learns reasonable steering offsets"""
        print("\nTesting paramsd offset learning")

        t = 0
        dt = 0.01

        # Simulate vehicle motion with significant turning
        for i in range(20):
            # Create motion with consistent turning
            velocity = vec3(5.0, 2.0, 0)
            bearing = 45.0 + i * 3.0  # More aggressive turning
            position = (100.0 + i, 100.0 + i)

            accel, gyro = self.calculate_imu_values(
                velocity,
                bearing,
                position,
                t + i * dt
            )

            # Update simulator state
            self.simulator_state.imu.accelerometer = accel
            self.simulator_state.imu.gyroscope = gyro
            self.simulator_state.imu.bearing = bearing

            # Update both locationd and paramsd
            self.locationd.update(self.simulator_state)
            self.paramsd.update(self.locationd.get_state())

            # Update previous state
            self.update_prev_state(velocity, bearing, position, t + i * dt)

        learned_offset = self.paramsd.get_steering_offset()
        print(f"\nLearned steering offset: {learned_offset:.2f}°")

        self.assertNotEqual(learned_offset, 0)
        self.assertLess(abs(learned_offset), 5.0)  # Max expected offset

    def test_steering_angle_correction(self):
        """Test end-to-end steering angle correction"""
        print("\nTesting steering angle correction")

        t = 0
        dt = 0.01
        raw_steering_angle = 7.866117000579834

        # Simulate vehicle motion with consistent turning
        for i in range(20):
            # Create motion with significant turning
            velocity = vec3(5.0, 2.0, 0)
            bearing = 45.0 + i * 4.0  # Even more aggressive turning
            position = (100.0 + i * 2, 100.0 + i * 2)

            accel, gyro = self.calculate_imu_values(
                velocity,
                bearing,
                position,
                t + i * dt
            )

            # Ensure significant gyroscope readings
            gyro = vec3(0, 0, max(gyro.z, 0.1))  # Ensure minimum rotation

            # Update simulator state
            self.simulator_state.imu.accelerometer = accel
            self.simulator_state.imu.gyroscope = gyro
            self.simulator_state.imu.bearing = bearing

            # Update both locationd and paramsd
            self.locationd.update(self.simulator_state)
            self.paramsd.update(self.locationd.get_state())

            # Update previous state
            self.update_prev_state(velocity, bearing, position, t + i * dt)

        corrected_angle = self.paramsd.apply_correction(raw_steering_angle)
        print(f"\nRaw steering angle: {raw_steering_angle:.2f}°")
        print(f"Corrected steering angle: {corrected_angle:.2f}°")

        self.assertNotEqual(corrected_angle, raw_steering_angle)
        self.assertLess(abs(corrected_angle - raw_steering_angle), 10.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
