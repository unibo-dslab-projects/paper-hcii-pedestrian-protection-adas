import carla
import cv2
import numpy as np
import pygame
import math
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt
import time
import random
import logging
from ultralytics.models.yolo import YOLO
from enum import Enum

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# MQTT setup
BROKER = "localhost"
PORT = 1883
TOPIC = "pedestrian_monitoring/"
mqtt_client = mqtt.Client()

try:
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    print(f"Listening to {TOPIC} on {BROKER}...")
except Exception as e:
    print("Connection failed:", e)

# CARLA setup
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# YOLO model
model = YOLO("yolov8n.pt")

# Global variables
steering_wheel = False
backwards_only_flag = False
braking_flag = False

class Mode(Enum):
    JOYSTICK = "joystick"
    KEYBOARD = "keyboard"

MODE = Mode.KEYBOARD

def spawn_camera(attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=1.2), carla.Rotation(pitch=-10)), fov=90.0, width=800, height=600, sensor_tick=0.0):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(fov))
    camera_bp.set_attribute('sensor_tick', str(sensor_tick))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

def remove_all(world: carla.World):
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('sensor.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.pedestrian.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.ai.walker'):
        a.destroy()

def spawn_walker(world: carla.World):
    blueprint_library = world.get_blueprint_library()
    walker_blueprints = list(blueprint_library.filter('walker.pedestrian.*'))

    if not walker_blueprints:
        print("No pedestrian blueprints available.")
        return None, None

    walker_bp = random.choice(walker_blueprints)

    if walker_bp.has_attribute("speed"):
        speed = random.uniform(0.8, 2.0)
        walker_bp.set_attribute('speed', str(speed))
    else:
        speed = 1.0

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Warning: No spawn points available for pedestrians. Cannot spawn walker.")
        return None, None

    spawn_point = random.choice(spawn_points)

    walker = world.try_spawn_actor(walker_bp, spawn_point)
    if walker is None:
        print("Failed to spawn walker at the chosen spawn point.")
        return None, None

    controller_bp = blueprint_library.find('controller.ai.walker')
    if controller_bp is None:
        print("No walker controller blueprint found.")
        return walker, None

    controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
    if controller:
        controller.start()
        def random_walk():
            while controller.is_alive and walker.is_alive:
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(speed)
                time.sleep(random.uniform(5, 15))
        threading.Thread(target=random_walk, daemon=True).start()

    return walker, controller

def send_warning(emergency):
    if emergency:
        mqtt_client.publish(TOPIC, "Pedestrian too close, emergency braking!")
    else:
        mqtt_client.publish(TOPIC, "Pedestrian approaching, slowing down the vehicle.")

def pedestrian_safety_monitoring(vehicle, results):
    velocity = vehicle.get_velocity()
    vehicle_speed = math.sqrt((velocity.x**2 + velocity.y**2 + velocity.z**2))

    for confidence, bbox, centroid in results:
        # In this merged version, we don't have depth information from the original script
        # So we'll just use the bounding box size as a proxy for distance
        bbox_width = bbox[2] - bbox[0]
        distance_estimate = 1000 / bbox_width  # Simple heuristic
        
        ttc = distance_estimate / vehicle_speed if vehicle_speed > 0 else float('inf')

        global backwards_only_flag
        global braking_flag

        if ttc < 1.0 and vehicle_speed > 0.0:
            control = vehicle.get_control()
            control.throttle = 0.0
            control.brake = 0.8
            vehicle.apply_control(control)
            send_warning(emergency=True)
            braking_flag = True
            print("Emergency braking")
            backwards_only_flag = True
        elif ttc < 2.0 and ttc > 1.0 and vehicle_speed > 0.0:
            control = vehicle.get_control()
            control.throttle = control.throttle / 3
            control.brake = 0.1
            vehicle.apply_control(control)
            send_warning(emergency=False)
            print("Soft emergency braking")
        elif distance_estimate >= 2.0:
            braking_flag = False
            backwards_only_flag = False

def pedestrian_detection(image):
    results = model(image)[0]
    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            if int(classes[i]) == 0 and confs[i] > 0.4:  # Class 0 is pedestrian in YOLO
                x1, y1, x2, y2 = boxes[i]
                bbox = (int(x1), int(y1), int(x2), int(y2))
                centroid = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                detections.append((confs[i], bbox, centroid))
    return detections

def process_image(image, res, vehicle):
    try:
        # Convert from BGRA to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Run pedestrian detection
        detections = pedestrian_detection(rgb_image)
        
        # Draw bounding boxes
        for conf, bbox, centroid in detections:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Run safety monitoring
        if vehicle is not None:
            pedestrian_safety_monitoring(vehicle, detections)
        
        res.put({'img': image, 'detections': detections})
    except Exception as e:
        print(f"Error processing image: {e}")

class GameLoop(object):
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.font.init()
        self.world = None
        self.original_settings = None
        self.fps = args.maxfps

        try:
            self.sim_world = client.get_world()
            if args.sync:
                self.original_settings = self.sim_world.get_settings()
                settings = self.sim_world.get_settings()
                if not settings.synchronous_mode:
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                self.sim_world.apply_settings(settings)

                traffic_manager = client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(True)

            if not self.sim_world.get_settings().synchronous_mode:
                print('WARNING: You are currently in asynchronous mode and could '
                    'experience some issues with the traffic simulation')

            self.display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0,0,0))
            pygame.display.flip()

            hud = mc.HUD(args.width, args.height)
            self.world = mc.World(self.sim_world, hud, args)
            self.controller = None
            if steering_wheel:
                self.controller = mc.DualControl(self.world)
            else:
                self.controller = mc.KeyboardControl()

            if args.sync:
                self.sim_world.tick()
            else:
                self.sim_world.wait_for_tick()
        except Exception:
            mc.logging.exception('Error creating the world')

    def get_speed(self, vehicle: carla.Vehicle):
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed

    def render(self, clock: pygame.time.Clock):
        self.world.tick(clock)
        self.world.render(self.display)
        pygame.display.flip()

    def get_blinkers_state(self):
        light_state = self.world.player.get_light_state()
        left_blinker = light_state & carla.VehicleLightState.LeftBlinker
        right_blinker = light_state & carla.VehicleLightState.RightBlinker
        return left_blinker, right_blinker

    def start(self, processed_output, autopilot=False, detection_center=100, threshold=20):
        self.world.player.set_autopilot(autopilot)
        try:
            clock = pygame.time.Clock()
            while True:
                if self.args.sync:
                    self.sim_world.tick()
                clock.tick_busy_loop(self.fps)

                if self.controller.parse_events(self.world, clock):
                    return

                # Show processed camera output
                try:
                    output = processed_output.get()
                    cv2.imshow('Processed image', output['img'])
                except Exception as e:
                    pass

                self.render(clock)
                cv2.waitKey(1)
        finally:
            if self.original_settings:
                self.sim_world.apply_settings(self.original_settings)
            if self.world is not None:
                self.world.destroy()
            pygame.quit()

def setup():
    argparser = mc.argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1", "2", "All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--maxfps',
        default=30,
        type=int,
        help='Fps of the client (default: 30)')
    args = argparser.parse_args()

    # Set max fps
    args.maxfps = 60

    # Set window resolution
    args.res = '1280x720'
    args.width, args.height = [int(x) for x in args.res.split('x')]

    # Set vehicle filter
    args.filter = 'vehicle.mercedes.coupe_2020'

    # Set synchronous mode
    args.sync = True

    log_level = mc.logging.DEBUG if args.debug else mc.logging.INFO
    mc.logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    mc.logging.info('listening to server %s:%s', args.host, args.port)

    print(mc.__doc__)

    return GameLoop(args)

# Main execution
if __name__ == '__main__':
    import threading
    import manual_control as mc
    from importlib import reload
    reload(mc)

    # Spawn pedestrians
    walkers = []
    controllers = []
    for _ in range(100):
        walker, controller = spawn_walker(world)
        if walker and controller:
            walkers.append(walker)
            controllers.append(controller)

    camera_width = 800
    camera_height = 600
    processed_output = Queue()

    # Setup the simulation environment
    game_loop = setup()

    # Get the vehicle and attach the camera
    vehicle = world.get_actors().filter('vehicle.*')[0]
    front_camera = spawn_camera(attach_to=vehicle, transform=carla.Transform(
            carla.Location(x=0.3, y=0.0, z=1.5),
            carla.Rotation(pitch=-10.0)
        ),
        width=camera_width, height=camera_height
    )

    cv2.namedWindow('Processed image', cv2.WINDOW_NORMAL)

    # Callback for the camera
    def camera_callback(image):
        try:
            video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            video_output = video_output[:, :, :3]
            with ThreadPoolExecutor() as executor:
                executor.submit(lambda: process_image(video_output, processed_output, vehicle))
        except Exception as e:
            print(e)

    # Attach the callback to the camera
    front_camera.listen(lambda image: camera_callback(image))

    def cleanup():
        remove_all(world)
        mqtt_client.disconnect()

    # Start the game loop
    try:
        game_loop.start(processed_output, autopilot=False, detection_center=100, threshold=7)
    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()