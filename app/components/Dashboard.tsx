import mqtt from "mqtt";
import { useEffect, useRef, useState } from "react";
import DangerMessage from "./DangerMessage";
import DisplaySettings, { AlertSettings } from "./DisplaySettings";
import Navbar from "./Navbar";
import NeutralMessage from "./NeutralMessage";
import WarningMessage from "./WarningMessage";

type Pedestrian = {
  x: number;
  distance: number;
  time_to_collision: number;
  camera_width: number;
}

enum PedestrianState {
  SAFE = "SAFE",
  WARNING = "WARNING",
  DANGER = "DANGER",
}

type CanvasSettings = {
  width: number;
  height: number;
  carWidth: number;
  carHeight: number;
  baseLaneWidth: number;
  farLaneWidth: number;
}

function Dashboard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const client = useRef<mqtt.MqttClient | null>(null);
  const [pedestrians, setPedestrians] = useState<Pedestrian[]>([]);

  const pedestrianImages = useRef<{ [color: string]: HTMLImageElement }>({});

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null);

  const canvasSettings: CanvasSettings = {
    width: 1200,
    height: 600,
    carWidth: 400,
    carHeight: 200,
    baseLaneWidth: 1200,
    farLaneWidth: 950,
  }
  const [alarmSettings, setAlarmSettings] = useState<AlertSettings>({
    soundOnWarning: true,
    soundOnDanger: true,
    flashing: true,
  });
  const [showWarning, setShowWarning] = useState<boolean>(false);

  function clearCanvas(ctx: CanvasRenderingContext2D) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "rgb(240, 240, 240)";
    ctx.fillRect(0, 0, canvasSettings.width, canvasSettings.height);
    drawLane(ctx);
  }

  function drawPedestrian(ctx: CanvasRenderingContext2D, pedestrian: Pedestrian) {
    const x = canvasSettings.width * (pedestrian.x / pedestrian.camera_width);
    const y = canvasSettings.height * (1 - pedestrian.distance / 30);

    let color = "white";
    const pedestrianState = calculatePedestrianState(pedestrian);
    if (pedestrianState === PedestrianState.DANGER) {
      color = "red";
    } else if (pedestrianState === PedestrianState.WARNING) {
      color = "yellow";
    }

    const img = pedestrianImages.current[color];

    const imgHeight = 150;
    const carBottom = canvasSettings.height - canvasSettings.carHeight;
    const adjustedY = x >= (canvasSettings.width - canvasSettings.carWidth - 40) / 2 && x <= (canvasSettings.width + canvasSettings.carWidth + 40) / 2
      ? Math.max(0, Math.min(y, carBottom - 120))
      : y;

    if (img.complete) {
      ctx.drawImage(img, x, adjustedY, imgHeight, imgHeight);
    } else {
      img.onload = () => {
        ctx.drawImage(img, x, adjustedY, imgHeight, imgHeight);
      };
    }
  }

  function drawLane(ctx: CanvasRenderingContext2D) {
    // Draw the lane as a simple gray trapezoid
    ctx.fillStyle = "gray";
    ctx.beginPath();
    ctx.setLineDash([]);
    ctx.moveTo((canvasSettings.width - canvasSettings.baseLaneWidth) / 2, canvasSettings.height);
    ctx.lineTo((canvasSettings.width + canvasSettings.baseLaneWidth) / 2, canvasSettings.height);
    ctx.lineTo((canvasSettings.width + canvasSettings.farLaneWidth) / 2, 0);
    ctx.lineTo((canvasSettings.width - canvasSettings.farLaneWidth) / 2, 0);
    ctx.closePath();
    ctx.fill();

    /* Draw the lane lines */
    ctx.strokeStyle = "white";
    ctx.lineWidth = 10;

    // Draw left curb line
    const curbOffset = 30;
    ctx.beginPath();
    ctx.moveTo((canvasSettings.width - canvasSettings.baseLaneWidth) / 2 + curbOffset, canvasSettings.height);
    ctx.lineTo((canvasSettings.width - canvasSettings.farLaneWidth) / 2 + curbOffset, 0);
    ctx.stroke();

    // Draw right curb line
    ctx.beginPath();
    ctx.moveTo((canvasSettings.width + canvasSettings.baseLaneWidth) / 2 - curbOffset, canvasSettings.height);
    ctx.lineTo((canvasSettings.width + canvasSettings.farLaneWidth) / 2 - curbOffset, 0);
    ctx.stroke();

    // Draw left diagonal lane line
    ctx.beginPath();
    ctx.moveTo((canvasSettings.width - canvasSettings.farLaneWidth) / 2 + canvasSettings.farLaneWidth - curbOffset, 0);
    ctx.lineTo((canvasSettings.width - canvasSettings.baseLaneWidth) / 2 + canvasSettings.baseLaneWidth - curbOffset, canvasSettings.height);
    ctx.stroke();
  }

  function calculatePedestrianState(pedestrian: Pedestrian): PedestrianState {
    const x = canvasSettings.width * (pedestrian.x / pedestrian.camera_width);
    const sectionWidth = canvasSettings.width / 6;

    if (x >= 2 * sectionWidth && x < 4 * sectionWidth && pedestrian.time_to_collision < 5) {
      return PedestrianState.DANGER;
    } else if (((x >= sectionWidth && x < 2 * sectionWidth) || (x >= 4 * sectionWidth && x < 5 * sectionWidth)) && pedestrian.time_to_collision < 10) {
      return PedestrianState.WARNING;
    }
    return PedestrianState.SAFE;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    client.current = mqtt.connect("ws://isi-simcar.campusfc.dir.unibo.it:8080");
    client.current.subscribe("pedestrian_monitoring");
    client.current.on("message", (topic, message) => {
      setPedestrians(() => {
        const newPedestrians = JSON.parse(message.toString()) as Pedestrian[];
        return newPedestrians;
      });
    });

    if (ctx) {
      clearCanvas(ctx);
    }

    const colors = ["white", "yellow", "red"];
    colors.forEach((color) => {
      const img = new Image();
      img.src = `/pedestrian-${color}.png`;
      pedestrianImages.current[color] = img;
    });

    // Initialize the AudioContext
    audioContextRef.current = new window.AudioContext();

    // Load the audio file into the AudioContext
    const loadAudio = async () => {
      const response = await fetch('/MobitasAlertSound.m4a');
      const audioData = await response.arrayBuffer();
      const audioBuffer = await audioContextRef.current.decodeAudioData(audioData);
      audioBufferRef.current = audioBuffer;
    };

    loadAudio();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const warning = pedestrians.some((pedestrian) => {
      const pedestrianState = calculatePedestrianState(pedestrian);
      return pedestrianState === PedestrianState.DANGER;
    });
    setShowWarning(warning);
    if (ctx) {
      clearCanvas(ctx);
      pedestrians.forEach((pedestrian) => {
        drawPedestrian(ctx, pedestrian);
      });
      if (warning && alarmSettings.soundOnWarning) {
        if (audioContextRef.current && audioBufferRef.current && !audioSourceNodeRef.current) {
          const source = audioContextRef.current.createBufferSource();
          source.buffer = audioBufferRef.current;
          source.connect(audioContextRef.current.destination);
          source.start(); // Play the sound
          audioSourceNodeRef.current = source;
        }
      } else {
        if (audioSourceNodeRef.current) {
          audioSourceNodeRef.current.stop();
          audioSourceNodeRef.current = null; // Reset the source node
        }
      }
    }
  }, [pedestrians]);

  return (
    <>
      <Navbar>
        <ul className="font-medium flex flex-col p-4 md:p-0 mt-4 border border-gray-100 rounded-lg bg-gray-50 md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:border-0 md:bg-white dark:bg-gray-800 md:dark:bg-gray-900 dark:border-gray-700">
          <li>
            <DisplaySettings settings={alarmSettings} setSettings={setAlarmSettings} />
          </li>
        </ul>
      </Navbar>
      <div className="p-4 flex justify-center">
        <div className="relative">
          <canvas ref={canvasRef} id="canvas" width={canvasSettings.width} height={canvasSettings.height} className="border mb-6 rounded-lg"></canvas>
          <img
            src="/car.png"
            alt="Car"
            className="absolute z-10"
            style={{
              width: `${canvasSettings.carWidth}px`,
              height: `${canvasSettings.carHeight}px`,
              left: `${(canvasSettings.width - canvasSettings.carWidth) / 2}px`,
              top: `${canvasSettings.height - canvasSettings.carHeight}px`,
            }}
          />
          {showWarning && <WarningMessage distance={pedestrians[0]?.distance} timeToCollision={pedestrians[0]?.time_to_collision} />}
          {!showWarning && <NeutralMessage />}
          {false && <DangerMessage distance={pedestrians[0]?.distance} timeToCollision={pedestrians[0]?.time_to_collision} />}
        </div>
        {
          showWarning && alarmSettings.flashing &&
          <>
            <div className="absolute top-0 left-0 h-full shadow-2xl bg-red-400 shadow-red-500 w-30 animate-ping z-0"></div>
            <div className="absolute top-0 right-0 h-full shadow-2xl bg-red-400 shadow-red-500 w-30 animate-ping z-0"></div>
          </>
        }
      </div>
    </>
  );
}

export default Dashboard;
