import { useEffect, useRef, useState } from "react";
import mqtt from "mqtt";
import WarningMessage from "./WarningMessage";
import NeutralMessage from "./NeutralMessage";
import DangerMessage from "./DangerMessage";

type Pedestrian = {
  x: number;
  y: number;
  distance: number;
}

function Dashboard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const client = useRef(mqtt.connect("ws://localhost:8080"));
  const [pedestrians, setPedestrians] = useState<Pedestrian[]>([]);
  const width = 1200;
  const height = 600;
  const carWidth = 200;
  const carHeight = 200;
  const baseLaneWidth = 800;
  const farLaneWidth = 400;

  function clearCanvas(ctx: CanvasRenderingContext2D) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "rgb(240, 240, 240)";
    ctx.fillRect(0, 0, width, height);
    drawCar(ctx);
    drawLane(ctx);
  }

  function drawCar(ctx: CanvasRenderingContext2D) {
    const carImage = new Image();
    carImage.src = "/car.png";
    carImage.onload = () => {
      ctx.drawImage(carImage, (width - carWidth) / 2 + baseLaneWidth * 0.25, height - carHeight, carWidth, carHeight);
    };
  }

  function drawPedestrian(ctx: CanvasRenderingContext2D, pedestrian: Pedestrian) {
    const pedestrianImage = new Image();
    pedestrianImage.src = "/pedestrian.png";
    pedestrianImage.onload = () => {
      ctx.beginPath();
      ctx.fillStyle = "rgba(150, 255, 0, 0.9)";
      ctx.arc(pedestrian.x + 25, pedestrian.y + 40, 50, 0, Math.PI * 2);
      ctx.fill();
      ctx.drawImage(pedestrianImage, pedestrian.x, pedestrian.y, 50, 50);
      ctx.font = "20px Arial";
      ctx.fillStyle = "black";
      ctx.fillText(`${pedestrian.distance} m`, pedestrian.x-10, pedestrian.y + 70);
    };
  }

  function drawLane(ctx: CanvasRenderingContext2D) {
    // Draw the lane as a simple gray trapezoid
    ctx.fillStyle = "gray";
    ctx.beginPath();
    ctx.setLineDash([]);
    ctx.moveTo((width - baseLaneWidth) / 2, height);
    ctx.lineTo((width + baseLaneWidth) / 2, height);
    ctx.lineTo((width + farLaneWidth) / 2, 0);
    ctx.lineTo((width - farLaneWidth) / 2, 0);
    ctx.closePath();
    ctx.fill();

    /* Draw the lane lines */
    ctx.strokeStyle = "white";
    ctx.lineWidth = 10;
    
    // Draw left curb line
    const curbOffset = 30;
    ctx.beginPath();
    ctx.moveTo((width - baseLaneWidth) / 2 + curbOffset, height);
    ctx.lineTo((width - farLaneWidth) / 2 + curbOffset, 0);
    ctx.stroke();

    // Draw right curb line
    ctx.beginPath();
    ctx.moveTo((width + baseLaneWidth) / 2 - curbOffset, height);
    ctx.lineTo((width + farLaneWidth) / 2 - curbOffset, 0);
    ctx.stroke();

    // Draw left diagonal lane line
    ctx.beginPath();
    ctx.moveTo((width - farLaneWidth) / 2 + farLaneWidth - curbOffset, 0);
    ctx.lineTo((width - baseLaneWidth) / 2 + baseLaneWidth - curbOffset, height);
    ctx.stroke();

    // Draw dashed center line
    ctx.beginPath();
    ctx.lineWidth = 5;
    ctx.setLineDash([20, 40]);
    ctx.moveTo(width / 2, height);
    ctx.lineTo(width / 2, 0);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    client.current.subscribe("hello");
    client.current.on("message", (topic, message) => {
      console.log(topic, message.toString());
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
      if (pedestrians.length > 0) {
        audioRef.current.play();
      }
    }
  }, [pedestrians]);

  return (
    <div className="relative">
      <audio ref={audioRef} id="audio">
        <source src="/MobitasAlertSound.m4a" type="audio/mp4"/>
      </audio>
      <canvas ref={canvasRef} id="canvas" width={width} height={height} className="border mb-6 rounded-lg"></canvas>
      <WarningMessage distance={pedestrians[0]?.distance} timeToCollision={pedestrians[0]?.distance / 2} />  
      {false && <NeutralMessage />}
      {false && <DangerMessage distance={pedestrians[0]?.distance} timeToCollision={pedestrians[0]?.distance / 2} />}
    </div>
  );
}

export default Dashboard;
