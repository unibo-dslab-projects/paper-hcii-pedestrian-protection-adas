function DangerMessage({
    distance,
    timeToCollision,
})  {
    return (
        <div className="text-center">
        <div className="bg-red-400 rounded-full text-center mb-4 p-5">
          <h1 className="text-4xl font-bold">EMERGENCY BRAKE!</h1>
          <p className="text-lg">Pedestrian detected ahead</p>
          <p className="text-8xl font-bold">{distance} m</p>
          <p className="text-4xl">Estimating <strong>{timeToCollision}s</strong> to collision</p>
        </div>
      </div>
    )
}

export default DangerMessage;
