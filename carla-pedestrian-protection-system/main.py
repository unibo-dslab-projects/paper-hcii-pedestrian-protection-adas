import carla, time, pygame, cv2, math, random, os, traceback, logging
import paho.mqtt.client as mqtt
import numpy as np
import tkinter as tk
from ultralytics.models.yolo import YOLO

logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ### Global Variables and Constants
# This cell defines global constants and settings that will be used throughout the notebook:
# - `BROKER`, `PORT`, and `TOPIC` define the MQTT broker settings.
# - `VIEW_WIDTH`, and `VIEW_HEIGHT` are determined using Tkinter to adapt the camera view size to the screen.

# In[ ]:


BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "pedestrian_monitoring/"

VIEW_WIDTH = tk.Tk().winfo_screenwidth()
VIEW_HEIGHT = tk.Tk().winfo_screenheight()
VIEW_FOV = 90


# ### CARLA and MQTT Setup
# In this cell, we set up the connection to the CARLA simulator and configure the MQTT client:
# - The MQTT client is initialized and connected to a public broker (`test.mosquitto.org`) on port 1883.
# - We attempt to connect to the CARLA server running on `localhost` at port 2000 and retrieve the simulation world.

# In[ ]:


## Carla set up
mqtt_client = mqtt.Client()

try:
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    print(f"Listening to {TOPIC} on {BROKER}...")
except Exception as e:
    print("Connection failed:", e)

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
except Exception as e:
    print(f"Failed to connect to CARLA: {e}")
    exit(1)

if not all(os.path.exists(f) for f in ["yolov8n.pt"]):
    print("YOLO model files not found!")
    exit(1)

world = client.get_world()
spectator = world.get_spectator()
capture = True
stored_image = None
stored_rgb_image = None
stored_depth_image = None
backwards_only_flag = False
braking_flag = False


# ### Weather Conditions Setup
# This cell defines several weather scenarios using CARLA's `WeatherParameters`. We create settings for:
# - Nighttime,
# - Sunny,
# - Rainy, and
# - Foggy conditions.
# These definitions allow us to test system performance under various environmental conditions.
# 

# In[ ]:


### weather choice
night_scenario = carla.WeatherParameters(
    sun_azimuth_angle=180.0,
    sun_altitude_angle=-90.0,
)

sunny_weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=5.0,
    sun_altitude_angle=70.0
)

rainy_weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=100.0,
    precipitation_deposits=80.0,
    wind_intensity=30.0,
    sun_altitude_angle=20.0
)

foggy_weather = carla.WeatherParameters(
    cloudiness=70.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=10.0,
    fog_density=80.0,
    fog_distance=10.0,
    fog_falloff=2.0,
    sun_altitude_angle=20.0
)

world.set_weather(sunny_weather)


# ### Image Processing Functions
# Here we define functions to process the raw sensor images:
# - `render(image)`: Converts the raw image data (which is in BGRA format) into an RGB NumPy array suitable for display.
# - `image_processing(image, target_size)`: Resizes and pads images to match the required input size for the YOLO model.
# - `set_rgb_image(image)` and `set_depth_image(image)`: Callback functions to store the latest RGB and Depth images.
# - `get_depth_at_pixel(x, y)`: Retrieves the depth value (in meters) for a specific pixel by decoding the 24-bit depth information from the BGRA image.
# 

# In[ ]:


### image processing

def render(image):
    """
    Converts a raw image from a sensor to a NumPy array suitable for display.

    Args:
        image: An image object with raw_data, height, and width attributes.

    Returns:
        A NumPy array representing the image in RGB format.
    """
    if image is not None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        return array

def image_processing(image, target_size):
    """
    Resizes and pads an image to fit the target size while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): The input image to be processed.
        target_size (tuple): The target size as a tuple (height, width).

    Returns:
        numpy.ndarray: The processed image, resized and padded to the target size, normalized to the range [0, 1].
    """
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    image_resized = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_LINEAR)

    image_padded = np.full((ih, iw, 3), 128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized

    image_padded = image_padded / 255.0
    return image_padded.astype(np.float32)

def set_rgb_image(image):
    """Stores the latest RGB image."""
    global stored_rgb_image
    stored_rgb_image = image

def set_depth_image(image):
    """Stores the latest Depth image."""
    global stored_depth_image
    stored_depth_image = image

def get_depth_at_pixel(x, y):
    """
    Retrieves the depth value (in meters) for a given pixel (x, y).

    Args:
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
    Returns:
        float: The depth value at the specified pixel in meters, or None if the depth image is not available.
    Raises:
        ValueError: If the pixel coordinates are out of bounds.
    """
    if stored_depth_image is None:
        return None

    if x < 0 or x >= stored_depth_image.width or y < 0 or y >= stored_depth_image.height:
        raise ValueError("Pixel coordinates are out of bounds.")

    depth_array = np.frombuffer(stored_depth_image.raw_data, dtype=np.uint8)
    depth_array = np.reshape(depth_array, (stored_depth_image.height, stored_depth_image.width, 4))

    blue  = depth_array[y, x, 0]
    green = depth_array[y, x, 1]
    red   = depth_array[y, x, 2]

    normalized_depth = (red + green * 256 + blue * 256**2) / (256**3 - 1)

    depth_in_meters = normalized_depth * 1000.0

    return depth_in_meters


# ### Simulation Environment Setup
# This cell defines functions to set up the CARLA simulation environment:
# - `setup_car()`: Spawns a vehicle in a random location and sets its light state based on the current weather.
# - `spawn_walker(world)`: Spawns a pedestrian (walker) and its controller in the simulation.
# - `setup_camera(car)`: Attaches both an RGB and a Depth camera to the vehicle at a defined transform, and configures their intrinsic calibration.
# - `move_spectator_to(transform, spectator, ...)`: Adjusts the spectator camera position so that the user has a clear view of the vehicle’s state.
# 

# In[ ]:


### setup simulation env

def setup_car():
    """
    Spawns an actor-vehicle in the simulation world and sets its light state based on the current weather conditions.
    Returns:
        carla.Actor: The spawned vehicle actor.
    """
    car_bp = world.get_blueprint_library().filter('vehicle.*')[0]
    location = random.choice(world.get_map().get_spawn_points())
    car = world.spawn_actor(car_bp, location)

    current_weather = world.get_weather()
    if current_weather.sun_altitude_angle < 20 or current_weather.fog_density > 40:
        lights = carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position
        car.set_light_state(carla.VehicleLightState(lights))
    else:
        car.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position))
    return car

def spawn_walker(world):
    """
    Spawns a pedestrian walker in the given CARLA world.
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
        destination = world.get_random_location_from_navigation()
        if destination:
            controller.go_to_location(destination)
        controller.set_max_speed(speed)

    return walker, controller

def setup_camera(car):
    """
    Spawns both an RGB and a Depth camera on the vehicle.

    Args:
        car (carla.Vehicle): The vehicle to which the cameras will be attached.
    Returns:
        tuple: A tuple containing the RGB camera and the Depth camera actors.
    """
    camera_transform = carla.Transform(carla.Location(x=1, y=-0.4, z=1.2), carla.Rotation())

    blueprint_library = world.get_blueprint_library()

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=car)
    rgb_camera.listen(lambda image: set_rgb_image(image))

    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
    depth_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=car)
    depth_camera.listen(lambda image: set_depth_image(image))

    calibration = np.identity(3)
    calibration[0, 2] = VIEW_WIDTH / 2.0
    calibration[1, 2] = VIEW_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    rgb_camera.calibration = calibration
    depth_camera.calibration = calibration

    return rgb_camera, depth_camera

def move_spectator_to(transform, spectator, distance=5.0, x=0, y=0, z=4, yaw=0, pitch=-30, roll=0):
    """
    Moves the spectator to a new position relative to the given transform.
    Args:
        transform (carla.Transform): The initial transform of the spectator.
        spectator (carla.Actor): The spectator actor to be moved.
        distance (float, optional): The distance to move the spectator backwards from the initial transform. Default is 5.0.
        x (float, optional): The offset to add to the x-coordinate of the new location. Default is 0.
        y (float, optional): The offset to add to the y-coordinate of the new location. Default is 0.
        z (float, optional): The offset to add to the z-coordinate of the new location. Default is 4.
        yaw (float, optional): The yaw rotation to add to the new transform. Default is 0.
        pitch (float, optional): The pitch rotation to set for the new transform. Default is -30.
        roll (float, optional): The roll rotation to set for the new transform. Default is 0.
    """
    back_location = transform.location - transform.get_forward_vector() * distance

    back_location.x += x
    back_location.y += y
    back_location.z += z
    transform.rotation.yaw += yaw
    transform.rotation.pitch = pitch
    transform.rotation.roll = roll

    spectator_transform = carla.Transform(back_location, transform.rotation)

    spectator.set_transform(spectator_transform)


# ### Vehicle Controller Functions
# This cell defines two functions for vehicle control:
# - `control_car(car, backwards_only=False)`: Uses keyboard input (via Pygame) to control the vehicle.
# - `control_car_with_wheel(car, joystick, backwards_only=False)`: Uses joystick input to control the vehicle.
# 
# Both functions ensure that the vehicle's behavior remains consistent regardless of the input method.
# 

# In[ ]:


### car controllers

def control_car(car, backwards_only=False):
    """
    Applies control to the main car based on pygame pressed keys.

    Args:
        car (carla.Vehicle): The car object to control.
        backwards_only (bool): If True, only allow backward movement and steering. Default is False.
    """
    control = car.get_control()

    control.brake = 0.0
    control.steer = 0.0

    move_spectator_to(car.get_transform(), spectator)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            key = event.key
            if backwards_only:
                if key == pygame.K_s:
                    control.reverse = True
                    control.throttle = 0.6
                if key == pygame.K_a:
                    control.throttle = 0.4
                    control.steer = -0.45
                if key == pygame.K_d:
                    control.throttle = 0.4
                    control.steer = 0.45
                if key == pygame.K_b:
                    control.brake = 0.7
                    control.throttle = 0.1
            else:
                if key == pygame.K_w:
                    control.reverse = False
                    control.throttle = 0.6
                if key == pygame.K_s:
                    control.reverse = True
                    control.throttle = 0.6
                if key == pygame.K_a:
                    control.throttle = 0.4
                    control.steer = -0.45
                if key == pygame.K_d:
                    control.throttle = 0.4
                    control.steer = 0.45
                if key == pygame.K_b:
                    control.brake = 0.7
                    control.throttle = 0.1

        car.apply_control(control)


def control_car_with_wheel(car, joystick, backwards_only=False):
    """
    Controls the car using a joystick input.
    Args:
        car (carla.Vehicle): The car object to control.
        joystick (pygame.joystick.Joystick): The joystick object to read inputs from.
        backwards_only (bool): If True, the car will only move in reverse. Default is False.
    """
    control = car.get_control()
    control.brake = 0.0
    control.steer = 0.0

    car.apply_control(control)

    pygame.event.pump()
    move_spectator_to(car.get_transform(), spectator)

    control.steer = max(-1.0, min(1.0, joystick.get_axis(0)))

    throttle = joystick.get_axis(3)
    if throttle >= 0.75:
        throttle = 0.75
    brake = joystick.get_axis(4)
    reverse_button = joystick.get_button(5)

    if backwards_only:
        control.reverse = True
        control.throttle = throttle
    else:
        if reverse_button:
            control.reverse = True
            control.throttle = throttle
        else:
            control.reverse = False
            control.throttle = throttle

    if brake > 0.1 and brake < 1.0:
        control.brake = brake

    car.apply_control(control)


# ### Pedestrian Safety Mechanism and Detection
# This cell defines the functions responsible for monitoring pedestrian safety and executing safety interventions:
# - `send_warning(emergency)`: Publishes a warning message via MQTT depending on whether an emergency braking scenario is detected.
# - `pedestrian_detection(image, model, vehicle, joystick)`: Processes the input RGB image using the YOLOv8 model to detect pedestrians. It filters detections based on confidence and passes the results to the safety monitoring function.
# - `pedestrian_safety_monitoring(vehicle, results, joystick)`: Computes vehicle speed, retrieves depth information, calculates the Time-to-Collision (TTC), and triggers soft or emergency braking based on TTC thresholds.

# In[ ]:


### pedestrian safety mechanism

def send_warning(emergency):
    """
    Sends a warning message via MQTT based on the emergency status.

    Args:
        emergency (bool): If True, sends an emergency braking message. If False, sends a slowing down message.
    """
    if emergency:
        mqtt_client.publish(TOPIC, "Pedestrian too close, emergency braking!")
    else:
        mqtt_client.publish(TOPIC, "Pedestrian approaching, slowing down the vehicle.")

def pedestrian_safety_monitoring(vehicle, results, joystick):
    """
    Monitors pedestrian safety using vehicle speed, braking distance, and depth camera data.

    Args:
        vehicle (Vehicle): The vehicle object to control.
        results (list): A list of tuples containing detection results, where each tuple consists of (confidence, bounding box, centroid).
        joystick (Joystick): The joystick object to control the vehicle.
    """
    velocity = vehicle.get_velocity()
    vehicle_speed = math.sqrt((velocity.x**2 + velocity.y**2 + velocity.z**2))

    for confidence, bbox, centroid in results:
        depth = get_depth_at_pixel(int(centroid[0]), int(centroid[1]))
        if depth is None or depth > 100:
            continue

        distance = depth

        ttc = distance / vehicle_speed

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
            # control_car(vehicle, backwards_only=True)
            control_car_with_wheel(vehicle, joystick, backwards_only=True)
            backwards_only_flag = True
        elif ttc < 2.0 and ttc > 1.0 and vehicle_speed > 0.0:
            control = vehicle.get_control()
            control.throttle = control.throttle / 3
            control.brake = 0.1
            vehicle.apply_control(control)
            send_warning(emergency=False)
            print("Soft emergency braking")
            # control_car(vehicle, backwards_only=False)
            control_car_with_wheel(vehicle, joystick, backwards_only=False)
        elif distance >= 2.0:
            # control_car(vehicle, backwards_only=False)
            control_car_with_wheel(vehicle, joystick, backwards_only=False)
            braking_flag = False
            backwards_only_flag = False

def pedestrian_detection(image, model, vehicle, joystick):
    """
    Detects pedestrians in an image using a given model and performs safety monitoring.

    Args:
        image (numpy.ndarray): The input image in which pedestrians are to be detected.
        model (object): The object detection model used to detect pedestrians.
        vehicle (object): The vehicle object that interacts with the safety monitoring system.
        joystick (object): The joystick object used for controlling the vehicle.

    Returns:
        list: A list of detections, where each detection is a tuple containing:
            - conf (float): The confidence score of the detection.
            - bbox (tuple): The bounding box coordinates of the detection (x1, y1, x2, y2).
            - centroid (tuple): The centroid coordinates of the bounding box (x, y).
    """

    results = model(image)[0]
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
    pedestrian_safety_monitoring(vehicle, detections, joystick)
    return detections


# ### Main Function and Running the Simulation
# This cell contains the main function, which orchestrates the simulation:
# - It spawns a vehicle, sets initial control, and moves the spectator.
# - Cameras are attached to the vehicle, and a number of pedestrians are spawned.
# - The main loop continuously ticks the CARLA world, processes images, runs pedestrian detection, and displays the results with bounding boxes.
# - Vehicle control is continuously updated based on user input and safety interventions.
# - The function handles a graceful exit on interruption.
# 

# In[ ]:


pygame.joystick.init() # only for joystick control
if pygame.joystick.get_count() == 0:
    print("No joystick detected! Please connect your Logitech steering wheel.")
    exit(1)
joystick = pygame.joystick.Joystick(0) # assuming only one steering wheel is connected
joystick.init()

def main():
    try:
        vehicle = setup_car()
        time.sleep(2)
        control = vehicle.get_control()
        control.brake = 0.0
        control.steer = 0.0
        vehicle.apply_control(control)
        move_spectator_to(vehicle.get_transform(), spectator)

        rgb_camera, depth_camera = setup_camera(vehicle)

        walkers = []
        controllers = []
        for _ in range(50):
            walker, controller = spawn_walker(world)
            if walker and controller:
                walkers.append(walker)
                controllers.append(controller)

        pygame.init()
        pygame.display.set_mode((200,200))

        display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

        model = YOLO("yolov8n.pt")

        while True:
            world.tick()
            control = vehicle.get_control()
            if control.throttle == -1.0:
                control.brake = 0.0
                control.throttle = 0.0
                vehicle.apply_control(control)
            while stored_rgb_image is None or stored_depth_image is None:
                world.tick()

            raw_image = render(stored_rgb_image)
            raw_image = raw_image.copy()

            bgr_for_display = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

            detections = pedestrian_detection(raw_image, model, vehicle, joystick)

            for conf, bbox, centroid in detections:
                cv2.rectangle(bgr_for_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

            cv2.imshow("Detection", bgr_for_display)


            pygame.display.flip()
            control_car_with_wheel(vehicle, joystick, backwards_only=backwards_only_flag)
            # control_car(vehicle, backwards_only=backwards_only_flag)
    except KeyboardInterrupt:
        print('\nSimulation interrupted by user')

    finally:
        print('Cleaning up...')
        pygame.quit()
        cv2.destroyAllWindows()
        mqtt_client.disconnect()
        actors = world.get_actors()
        if 'rgb_camera' in locals():
            rgb_camera.destroy()
        if 'depth_camera' in locals():
            depth_camera.destroy()
        actors = world.get_actors()
        for actor in actors:
            if isinstance(actor, (carla.Vehicle, carla.Walker)):
                actor.destroy()
        print('Done.')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f'Exception occurred: {e}')


# ### Cleanup
# In case something goes wrong with the graceful exit, this cell can be run to clean up the simulation environment from all the actors.

# In[ ]:


actors = world.get_actors()
for actor in actors:
    if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
        actor.destroy()

