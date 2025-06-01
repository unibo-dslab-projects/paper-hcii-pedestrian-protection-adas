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
  const audioRef = useRef<HTMLAudioElement>(null);
  const client = useRef(mqtt.connect("ws://localhost:8080"));
  const [pedestrians, setPedestrians] = useState<Pedestrian[]>([]);
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
  const showWarning: boolean = pedestrians.length > 0

  function clearCanvas(ctx: CanvasRenderingContext2D) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "rgb(240, 240, 240)";
    ctx.fillRect(0, 0, canvasSettings.width, canvasSettings.height);
    drawCar(ctx);
    drawLane(ctx);
  }

  function drawCar(ctx: CanvasRenderingContext2D) {
    const carImage = new Image();
    carImage.src = "/car.png";
    carImage.onload = () => {
      ctx.drawImage(carImage, (canvasSettings.width - canvasSettings.carWidth) / 2, canvasSettings.height - canvasSettings.carHeight, canvasSettings.carWidth, canvasSettings.carHeight);
    };
  }

  function drawPedestrian(ctx: CanvasRenderingContext2D, pedestrian: Pedestrian) {
    const pedestrianImage = new Image();
    pedestrianImage.src = "/pedestrian.png";
    const pedestrianAtX = canvasSettings.width * (pedestrian.x / pedestrian.camera_width);
    const pedestrianAtY = canvasSettings.height * (1 - pedestrian.distance / 30);
    pedestrianImage.onload = () => {
      ctx.drawImage(pedestrianImage, pedestrianAtX, pedestrianAtY, 150, 150);
    };
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

    // Draw dashed center line
    /*ctx.beginPath();
    ctx.lineWidth = 5;
    ctx.setLineDash([20, 40]);
    ctx.moveTo(canvasSettings.width / 2, canvasSettings.height);
    ctx.lineTo(canvasSettings.width / 2, 0);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash */
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
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
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) {
      clearCanvas(ctx);
      pedestrians.forEach((pedestrian) => {
        drawPedestrian(ctx, pedestrian);
      });
      if (pedestrians.length > 0 && alarmSettings.soundOnWarning) {
        if (audioRef.current.paused) {
          audioRef.current.currentTime = 0;
        }
        audioRef.current.play();
      } else {
        audioRef.current.pause();
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
          <audio ref={audioRef} id="audio" loop>
            <source src="/MobitasAlertSound.m4a" type="audio/mp4" />
          </audio>
          <canvas ref={canvasRef} id="canvas" width={canvasSettings.width} height={canvasSettings.height} className="border mb-6 rounded-lg"></canvas>
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
