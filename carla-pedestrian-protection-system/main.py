import carla
import cv2
import numpy as np
import pygame
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass
from typing import List
import random
import time
import threading
import json

#########################################
################ Classes ################
#########################################

@dataclass
class Pedestrian:
    x: int
    y: int
    distance: float # maybe int with fixed precision is enough
    time_to_collision: float

class Mode(Enum):
    KEYBOARD = 1
    STEERING_WHEEL = 2

##########################################
################# Config #################
##########################################

MODE = Mode.STEERING_WHEEL
CAMERA_DEBUG = False
NUM_WALKERS = 50

BROKER = "localhost"
PORT = 1883
TOPIC = "pedestrian_monitoring"

CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 720
VIEW_FOV = 120

model = YOLO("yolov8n.pt")

if CAMERA_DEBUG:
    cv2.namedWindow('RGB image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth image', cv2.WINDOW_NORMAL)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
spectator = world.get_spectator()

if MODE == Mode.STEERING_WHEEL:
    import manual_control_steeringwheel as mc
    from importlib import reload
    reload(mc)
elif MODE == Mode.KEYBOARD:
    import manual_control as mc
    from importlib import reload
    reload(mc)
else:
    raise ValueError("Invalid mode selected. Choose either MODE.KEYBOARD or MODE.STEERING_WHEEL.")

input_rgb_image = None
input_rgb_image_lock = threading.Lock()

input_depth_image = None
input_depth_image_lock = threading.Lock()

processed_output = None
processed_output_lock = threading.Lock()

mqtt_client = mqtt.Client()

try:
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    print(f"Listening to {TOPIC} on {BROKER}...")
except Exception as e:
    print("Connection failed:", e)

###########################################
############ Utility functions ############
###########################################

def remove_all(world: carla.World):
    """
    Removes all actors from the CARLA world, including vehicles, sensors, and pedestrians.
    Args:
        world (carla.World): The CARLA world object from which actors will be removed.
    """
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('sensor.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.pedestrian.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.ai.walker'):
        a.destroy()

def setup_camera(car: carla.Vehicle):
    """
    Spawns both an RGB and a Depth camera on the vehicle.

    Args:
        car (carla.Vehicle): The vehicle to which the cameras will be attached.
    Returns:
        tuple: A tuple containing the RGB camera and the Depth camera actors.
    """
    # camera_transform = carla.Transform(carla.Location(x=1, y=-0.4, z=1.2), carla.Rotation())
    camera_transform = carla.Transform(carla.Location(x=1, y=0, z=1.2), carla.Rotation())

    blueprint_library = world.get_blueprint_library()

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=car)

    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    depth_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=car)

    calibration = np.identity(3)
    calibration[0, 2] = CAMERA_WIDTH / 2.0
    calibration[1, 2] = CAMERA_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = CAMERA_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    rgb_camera.calibration = calibration
    depth_camera.calibration = calibration

    return rgb_camera, depth_camera

def spawn_walker(world: carla.World):
    """
    Spawns a pedestrian walker in the given CARLA world and makes it walk to a random destination.
    Args:
        world (carla.World): The CARLA world object where the walker will be spawned.
    Returns:
        tuple: A tuple containing the walker actor and its controller actor.
               If spawning fails, returns (None, None).
    """
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
        # Make the walker walk randomly by continuously assigning new random destinations
        def random_walk():
            while controller.is_alive and walker.is_alive:
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(speed)
                # Wait for a random time before assigning a new destination
                time.sleep(random.uniform(5, 15))
        threading.Thread(target=random_walk, daemon=True).start()

    return walker, controller

def send_pedestrian_data(pedestrians: List[Pedestrian]):
    """
    Sends the detected pedestrian data to the MQTT broker.
    
    Args:
        pedestrians (list): A list of Pedestrian objects containing their positions and distances.
    """

    payload = [
        {
            "x": ped.x,
            "distance": f"{ped.distance:.2f}",
            "camera_width": CAMERA_WIDTH,
            "time_to_collision": f"{ped.time_to_collision:.2f}",
        } for ped in pedestrians
    ]

    try:
        mqtt_client.publish(TOPIC, json.dumps(payload))
    except Exception as e:
        print(f"Failed to publish data: {e}")

##########################################
############ Image Processing ############
##########################################

def detect_pedestrians(image):
    results = model.predict(image, device='cpu', verbose=False)[0] # device='cuda:0' for GPU inference
    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            if int(classes[i]) == 0 and confs[i] > 0.4:
                x1, y1, x2, y2 = boxes[i]
                bbox = (int(x1), int(y1), int(x2), int(y2))
                centroid = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                detections.append((confs[i], bbox, centroid))
    return detections

def get_distance_to_pedestrian_centroid(centroid, depth_image):
    x, y = centroid

    blue  = depth_image[y, x, 0]
    green = depth_image[y, x, 1]
    red   = depth_image[y, x, 2]

    normalized_depth = (red + green * 256 + blue * 256**2) / (256**3 - 1)

    depth_in_meters = normalized_depth * 1000.0

    # print(f"Depth at pixel ({x}, {y}): {depth_in_meters:.2f} meters")

    return depth_in_meters


def process_image():
    '''
    Process the input image, detect lanes, interpolate and draw the echidistant lane.
    If an error occurs, the original image with no relevations is put in the result queue.

    Args:
        image: the input image
        res: the queue to put the result
    '''
    global input_rgb_image, processed_output
    while True:
        with input_rgb_image_lock, input_depth_image_lock:
            if input_rgb_image is None or input_depth_image is None:
                continue
            rgb_image = input_rgb_image
            depth_image = input_depth_image
        
        rgb_image = np.reshape(np.copy(rgb_image.raw_data), (rgb_image.height, rgb_image.width, 4))
        rgb_image = rgb_image[:, :, :3]
        depth_image = np.reshape(np.copy(depth_image.raw_data), (depth_image.height, depth_image.width, 4))

        vehicle_speed = vehicle.get_velocity()
        vehicle_speed_mps = np.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2 + vehicle_speed.z**2)

        # print(f"Vehicle speed: {vehicle_speed_mps:.2f} m/s")

        detections = detect_pedestrians(rgb_image)
        detected_pedestrians: list[Pedestrian] = []
        for _, _, centroid in detections:
            distance = get_distance_to_pedestrian_centroid(centroid, depth_image)
            time_to_collision = distance / vehicle_speed_mps if vehicle_speed_mps > 0 else float('inf')
            detected_pedestrians.append(Pedestrian(x=centroid[0], y=centroid[1], distance=distance, time_to_collision=time_to_collision))
        
        for pedestrian in detected_pedestrians:
            print(f"Pedestrian detected at ({pedestrian.x}, {pedestrian.y}) with distance {pedestrian.distance} meters")
        
        send_pedestrian_data(detected_pedestrians)

        with processed_output_lock:
            processed_output = {
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "detections": detections,
            }

##########################################
########### Gameloop and Setup ###########
##########################################

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
            if MODE == Mode.STEERING_WHEEL:
                self.controller = mc.DualControl(self.world)
            elif MODE == Mode.KEYBOARD:
                self.controller = mc.KeyboardControl()

            if args.sync:
                self.sim_world.tick()
            else:
                self.sim_world.wait_for_tick()
        except Exception:
            mc.logging.exception('Error creating the world')

    def render(self, clock: pygame.time.Clock):
        self.world.tick(clock)
        self.world.render(self.display)
        pygame.display.flip()

    def start(self):
        self.world.player.set_autopilot(False)
        try:
            clock = pygame.time.Clock()
            while True:
                if self.args.sync:
                    self.sim_world.tick()
                clock.tick_busy_loop(self.fps)

                if self.controller.parse_events(self.world, clock):
                    return

                try:

                    with processed_output_lock:
                        output_rgb_image = processed_output["rgb_image"]
                        output_depth_image = processed_output["depth_image"]
                        output_detections = processed_output["detections"]

                    bgr_for_display = cv2.cvtColor(output_rgb_image, cv2.COLOR_RGB2BGR)
                    depth_for_display = cv2.cvtColor(output_depth_image, cv2.COLOR_RGB2BGR)
                    for _, bbox, _ in output_detections:
                        cv2.rectangle(bgr_for_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

                    if CAMERA_DEBUG:
                        if bgr_for_display is not None:
                            cv2.imshow('RGB image', bgr_for_display)
                        
                        if depth_for_display is not None:
                            cv2.imshow('Depth image', depth_for_display)

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

    # Set window resolution
    args.res = '1280x720'
    args.width, args.height = CAMERA_WIDTH, CAMERA_HEIGHT #[int(x) for x in args.res.split('x')]

    # Set vehicle filter
    args.filter = 'vehicle.mercedes.coupe_2020'

    # Set synchronous mode
    args.sync = True

    log_level = mc.logging.DEBUG if args.debug else mc.logging.INFO
    mc.logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    mc.logging.info('listening to server %s:%s', args.host, args.port)

    print(mc.__doc__)

    return GameLoop(args)

##########################################
############ Simulation Setup ############
##########################################

for _ in range(NUM_WALKERS):
    _, _ = spawn_walker(world)

# setup the simulation environment
game_loop = setup()

# get the vehicle and attach the camera
vehicle: carla.Vehicle = world.get_actors().filter('vehicle.*')[0]
rgb_camera, depth_camera = setup_camera(vehicle)

threading.Thread(target=process_image, daemon=True).start()

def rgb_camera_callback(image):
    try:
        global input_rgb_image
        with input_rgb_image_lock:
            input_rgb_image = image
    except Exception as e:
        print(e.with_traceback())

def depth_camera_callback(image):
    try:
        global input_depth_image
        with input_depth_image_lock:
            input_depth_image = image
    except Exception as e:
        print(e.with_traceback())

rgb_camera.listen(lambda image: rgb_camera_callback(image))
depth_camera.listen(lambda image: depth_camera_callback(image))

def cleanup():
    remove_all(world)

# start the game loop
try:
    game_loop.start()
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()