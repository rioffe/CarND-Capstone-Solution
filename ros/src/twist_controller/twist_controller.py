from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, dbw_enabled,
                vehicle_mass, fuel_capacity,
                brake_deadband, decel_limit,
                accel_limit, wheel_radius,
                wheel_base, steer_ratio,
                max_lat_accel, max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        #self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        #self.wheel_base = wheel_base
        #self.steer_ratio = steer_ratio
        #self.max_lat_accel = max_lat_accel
        #self.max_steer_angle = max_steer_angle

        min_speed = 0.0
        self.yaw_controller = YawController(wheel_base, steer_ratio,
                                            min_speed, max_lat_accel,
                                            max_steer_angle)
        kp = 0.5
        ki = 1.0
        kd = 0.1
        self.pid_throttle = PID(kp, ki, kd, mn=0.0, mx=1.0)

    def control(self, dbw_enabled, goal_linear_v, goal_angular_v, current_linear_v, dt):
        if not dbw_enabled:
            self.pid_throttle.reset()

        else:
            error = goal_linear_v - current_linear_v
            throttle = self.pid_throttle.step(error, dt)
            brake = 0.0
            steer = self.yaw_controller.get_steering(goal_linear_v,
                                                goal_angular_v,
                                                current_linear_v)
        return throttle, brake, steer
