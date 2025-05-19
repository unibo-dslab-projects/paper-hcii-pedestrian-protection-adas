import { use, useEffect, useRef, useState } from "react";
import mqtt from "mqtt";

type Pedestrian = {
  x: number;
  y: number;
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

  function drawLaneLines(ctx: CanvasRenderingContext2D) {
    ctx.strokeStyle = "white";
    ctx.lineWidth = 10;
    ctx.setLineDash([]);
    ctx.beginPath();
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
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(pedestrian.x, pedestrian.y, 5, 0, Math.PI * 2);
        ctx.fillText(`(${pedestrian.x}, ${pedestrian.y})`, pedestrian.x + 10, pedestrian.y);
        ctx.fill();
        ctx.fillStyle = "black";
        ctx.lineTo(100, 100);
        ctx.stroke();
      });
    }
  }, [pedestrians]);

  return (
    <div>
      <canvas ref={canvasRef} id="canvas" width="800" height="600" className="border mb-6 rounded-lg">
      </canvas>
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
