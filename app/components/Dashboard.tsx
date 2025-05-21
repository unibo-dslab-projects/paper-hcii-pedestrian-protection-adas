import { use, useEffect, useRef, useState } from "react";
import mqtt from "mqtt";

type Pedestrian = {
  x: number;
  y: number;
  distance: number;
}

function Dashboard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const client = useRef(mqtt.connect("ws://localhost:8080"));
  const [pedestrians, setPedestrians] = useState<Pedestrian[]>([]);

  function clearCanvas(ctx: CanvasRenderingContext2D) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "gray";
    ctx.fillRect(0, 0, 800, 600);
    drawCar(ctx);
    drawLaneLines(ctx);
  }

  function drawCar(ctx: CanvasRenderingContext2D) {
    const carImage = new Image();
    carImage.src = "/car.png";
    carImage.onload = () => {
      ctx.drawImage(carImage, 420, 400, 200, 200);
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

  function drawLaneLines(ctx: CanvasRenderingContext2D) {
    ctx.strokeStyle = "white";
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.setLineDash([]);
    ctx.moveTo(150, 600);
    ctx.lineTo(200, 0);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(650, 600);
    ctx.lineTo(600, 0);
    ctx.stroke();

    ctx.beginPath();
    ctx.setLineDash([20, 20]);
    ctx.moveTo(400, 600);
    ctx.lineTo(400, 0);
    ctx.stroke();
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
    }
  }, [pedestrians]);

  return (
    <div>
     <canvas ref={canvasRef} id="canvas" width="1200" height="600" className="border mb-6 rounded-lg"></canvas>
      <div className="text-center">
        <div className="bg-yellow-400 rounded-full text-center mb-4 p-10">
          <h1 className="text-4xl font-bold">SLOW DOWN!</h1>
          <p className="text-lg">Pedestrian detected ahead</p>
          <p className="text-9xl font-bold">{pedestrians[0]?.distance} m</p>
          <p className="text-4xl">Estimating <strong>{pedestrians[0]?.distance / 2}s</strong> to collision</p>
        </div>
      </div>
      <h1 className="text-2xl font-bold mt-4">Dashboard</h1>
      <ul>
        {pedestrians.map((pedestrian, index) => (
          <li key={index} className="text-lg">
            Pedestrian {index + 1}: x: {pedestrian.x}, y: {pedestrian.y}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Dashboard;
