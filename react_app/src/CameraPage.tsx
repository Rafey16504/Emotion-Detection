import React, { useEffect, useState } from "react";
import io from "socket.io-client";

const CameraPage: React.FC = () => {
  const [frame, setFrame] = useState<string | null>(null);
  const [emotionData, setEmotionData] = useState<{
    emotion: string;
    confidence: number;
  } | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);

  useEffect(() => {
    const socket = io("http://localhost:5000");

    socket.on(
      "update",
      (data: {
        frame: string;
        emotion: { emotion: string; confidence: number };
      }) => {
        console.log("Received update:", data);
        setFrame(data.frame);
        setEmotionData(data.emotion);
      }
    );

    return () => {
      socket.disconnect();
    };
  }, []);

  const startCamera = () => {
    const socket = io("http://localhost:5000");
    socket.emit("start_camera");
    setCameraActive(true);
  };

  const stopCamera = () => {
    const socket = io("http://localhost:5000");
    socket.emit("stop_camera");
    setCameraActive(false);
    setFrame(null);
    setEmotionData(null);
  };

  return (
    <div className="flex flex-col justify-center items-center min-h-screen bg-gradient-to-r from-blue-500 to-purple-500 text-white">
      <div className="flex hover:text-red-700 transition duration-300">
        
        <a
          href="/"
          className=""
        >
          
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0l-2-2m2 2V4a1 1 0 011-1h3m-3 4l2-2"
            />
            
          </svg>
          
        </a>
        <a href="/" className=" text-lg">
          Home
          </a>
      </div>
      <h1 className="text-3xl underline mb-6 font-semibold">
        Real-Time Emotion Detection
      </h1>
      
      <div className="mb-6">
        {!cameraActive ? (
          <button
            onClick={startCamera}
            className="bg-white text-blue-500 px-6 py-3 rounded-full font-semibold hover:bg-gray-100 transition duration-300"
          >
            Enable Camera
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="bg-red-500 text-white px-6 py-3 rounded-full font-semibold hover:bg-red-600 transition duration-300"
          >
            Disable Camera
          </button>
        )}
      </div>

      <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-2xl text-black">
        {cameraActive && frame ? (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt="Camera Feed"
            className="rounded-lg shadow-md mb-4"
          />
        ) : (
          <p className="text-red-500 text-center">Camera is not active.</p>
        )}

        {cameraActive && emotionData ? (
          <div className="mt-4">
            <p className="text-lg font-semibold text-center">
              Detected Emotion:
            </p>
            <p className="text-3xl text-blue-600 font-bold text-center">
              {emotionData.emotion}
            </p>
            {/* <p className="text-sm mt-2 text-center">Confidence: {(emotionData.confidence * 100).toFixed(2)}%</p> */}
          </div>
        ) : (
          <p className="text-red-500 text-center"></p>
        )}
      </div>
    </div>
  );
};

export default CameraPage;
