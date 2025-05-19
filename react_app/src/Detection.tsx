import { Link } from "react-router-dom";
import { useState, useEffect, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import io from "socket.io-client";

const Detection = () => {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [predictions, setPredictions] = useState<any[]>([]);
  const webcamRef = useRef<Webcam>(null);
  const socketRef = useRef<any>(null);
  const [videoSize, setVideoSize] = useState({ width: 640, height: 480 });

  useEffect(() => {
    const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
    console.log("Connecting to backend at:", backendUrl);
    socketRef.current = io(backendUrl, {
      transports: ["websocket"],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketRef.current.on("predictions", (data: any) => {
      setPredictions(data);
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  const toggleCamera = useCallback(() => {
    setIsCameraOn((prev) => !prev);
  }, []);

  const captureFrame = useCallback(() => {
    if (webcamRef.current && isCameraOn) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        socketRef.current.emit("process_frame", {
          image: imageSrc.split(",")[1],
        });
      }
    }
  }, [isCameraOn]);

  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;

    if (isCameraOn) {
      interval = setInterval(captureFrame, 100);
    } else {
      clearInterval(interval);
    }

    return () => clearInterval(interval);
  }, [isCameraOn, captureFrame]);

  useEffect(() => {
    if (webcamRef.current && webcamRef.current.video) {
      const video = webcamRef.current.video;
      const observer = new ResizeObserver((entries) => {
        const { width, height } = entries[0].contentRect;
        setVideoSize({ width, height });
      });
      observer.observe(video);
      return () => observer.disconnect();
    }
  }, [isCameraOn]);

  const getScaledPrediction = (bbox: number[]) => {
    const scaleX = videoSize.width / 640;
    const scaleY = videoSize.height / 480;

    const isSmallScreen = window.innerWidth < 768;
    const sizeMultiplier = isSmallScreen ? 1.5 : 1.0;

    const scaledBbox = {
      left: bbox[0] * scaleX,
      top: bbox[1] * scaleY,
      width: bbox[2] * scaleX * sizeMultiplier,
      height: bbox[3] * scaleY * sizeMultiplier,
    };

    return {
      left: Math.max(
        0,
        Math.min(scaledBbox.left, videoSize.width - scaledBbox.width)
      ),
      top: Math.max(
        0,
        Math.min(scaledBbox.top, videoSize.height - scaledBbox.height)
      ),
      width: Math.min(scaledBbox.width, videoSize.width),
      height: Math.min(scaledBbox.height, videoSize.height),
    };
  };

  return (
    <div className="flex flex-col justify-center items-center min-h-screen bg-gray-900">
      <header className="absolute top-4 left-4 z-10">
        <Link
          to="/"
          className="flex items-center space-x-2 text-white hover:text-purple-400 transition"
        >
          <svg
            className="h-6 w-6"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
            <g
              id="SVGRepo_tracerCarrier"
              stroke-linecap="round"
              stroke-linejoin="round"
            ></g>
            <g id="SVGRepo_iconCarrier">
              {" "}
              <path
                d="M2 12.2039C2 9.91549 2 8.77128 2.5192 7.82274C3.0384 6.87421 3.98695 6.28551 5.88403 5.10813L7.88403 3.86687C9.88939 2.62229 10.8921 2 12 2C13.1079 2 14.1106 2.62229 16.116 3.86687L18.116 5.10812C20.0131 6.28551 20.9616 6.87421 21.4808 7.82274C22 8.77128 22 9.91549 22 12.2039V13.725C22 17.6258 22 19.5763 20.8284 20.7881C19.6569 22 17.7712 22 14 22H10C6.22876 22 4.34315 22 3.17157 20.7881C2 19.5763 2 17.6258 2 13.725V12.2039Z"
                stroke="#ffffff"
                stroke-width="1.5"
              ></path>{" "}
              <path
                d="M15 18H9"
                stroke="#ffffff"
                stroke-width="1.5"
                stroke-linecap="round"
              ></path>{" "}
            </g>
          </svg>
          <span className="text-lg font-semibold">Home</span>
        </Link>
      </header>

      {/* Camera Feed and Prediction Box */}
      <div className="flex flex-col items-center justify-center w-full max-w-screen-lg p-4">
        <div className="relative w-full max-w-3xl">
          {isCameraOn ? (
            <>
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={{ facingMode: "user" }}
                className="rounded-3xl shadow-2xl w-full h-auto border-4 border-purple-500/20"
              />

              <div className="absolute lg:-inset-10 inset-16 pointer-events-none">
                {predictions.map((pred, index) => {
                  const scaledBbox = getScaledPrediction(pred.bbox);

                  return (
                    <div
                      key={index}
                      className="absolute border-2 border-green-500 rounded-md"
                      style={{
                        left: scaledBbox.left,
                        top: scaledBbox.top,
                        width: scaledBbox.width,
                        height: scaledBbox.height,
                      }}
                    >
                      <div className="bg-red-500 text-white text-xs px-2 py-1 rounded-b absolute -bottom-8 left-1/2 transform -translate-x-1/2">
                        {`${pred.emotion}`}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center text-center text-white">
              <svg
                className="lg:h-96 lg:w-96 h-72 w-72"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                <g
                  id="SVGRepo_tracerCarrier"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                ></g>
                <g id="SVGRepo_iconCarrier">
                  {" "}
                  <path
                    d="M2 18C2 19.8856 2 20.8284 2.58579 21.4142C3.17157 22 4.11438 22 6 22C7.88562 22 8.82843 22 9.41421 21.4142C10 20.8284 10 19.8856 10 18V6C10 4.11438 10 3.17157 9.41421 2.58579C8.82843 2 7.88562 2 6 2C4.11438 2 3.17157 2 2.58579 2.58579C2 3.17157 2 4.11438 2 6V14"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  ></path>{" "}
                  <path
                    d="M22 6C22 4.11438 22 3.17157 21.4142 2.58579C20.8284 2 19.8856 2 18 2C16.1144 2 15.1716 2 14.5858 2.58579C14 3.17157 14 4.11438 14 6V18C14 19.8856 14 20.8284 14.5858 21.4142C15.1716 22 16.1144 22 18 22C19.8856 22 20.8284 22 21.4142 21.4142C22 20.8284 22 19.8856 22 18V10"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  ></path>{" "}
                </g>
              </svg>
              <h3 className="text-2xl font-semibold">Camera Paused</h3>
              <p className="text-gray-400">Click below to start detection</p>
            </div>
          )}
        </div>
      </div>

      {/* Control Button */}
      <div className="mt-6 p-4 flex flex-col items-center space-y-4">
        <button
          className={`flex items-center space-x-3 px-6 py-3 rounded-full transition-all duration-300 shadow-xl ${
            isCameraOn
              ? "bg-red-500 hover:bg-red-600"
              : "bg-green-500 hover:bg-green-600"
          }`}
          onClick={toggleCamera}
        >
          {isCameraOn ? (
            <>
              <svg
                className="h-6 w-6"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                <g
                  id="SVGRepo_tracerCarrier"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                ></g>
                <g id="SVGRepo_iconCarrier">
                  {" "}
                  <path
                    d="M7.9 16.0999C7.31233 15.266 6.99789 14.2702 7.00001 13.25C6.99841 12.3353 7.24772 11.4378 7.72081 10.655C8.19389 9.87222 8.87261 9.23421 9.68312 8.81039C10.4936 8.38658 11.4049 8.19321 12.3176 8.25131C13.2304 8.30942 14.1098 8.61682 14.86 9.13998"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                  <path
                    d="M16.6991 11.5399C16.9013 12.0872 17.003 12.6666 16.9991 13.25C16.9991 14.5761 16.4723 15.8478 15.5346 16.7855C14.5969 17.7232 13.3252 18.25 11.9991 18.25C11.4162 18.2481 10.8378 18.1466 10.2891 17.95"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                  <path
                    d="M20.9486 7.28992C21.626 8.02965 22.0008 8.99697 21.9986 10L21.3286 18C21.2211 19.081 20.7211 20.0851 19.9232 20.8223C19.1252 21.5594 18.0847 21.9784 16.9986 22H6.9986C6.76728 22.0003 6.53638 21.9802 6.30859 21.9399"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                  <path
                    d="M3.63874 20.36C3.07236 19.6977 2.72326 18.8773 2.63874 18.01L1.96876 10.01C1.96722 9.14153 2.24838 8.29613 2.76975 7.60156C3.29113 6.907 4.02437 6.40091 4.85874 6.15991C5.68459 5.9183 6.41052 5.41689 6.92875 4.72998L7.76874 3.60999C8.14133 3.1132 8.62446 2.70996 9.17988 2.43225C9.7353 2.15454 10.3478 2.01001 10.9688 2.01001H12.9688C13.5897 2.01001 14.2022 2.15454 14.7576 2.43225C15.313 2.70996 15.7962 3.1132 16.1687 3.60999L17.0187 4.72998C17.3398 5.14029 17.7413 5.4806 18.1987 5.72998"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                  <path
                    d="M22 2L2 22"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                </g>
              </svg>
              <span>Stop Camera</span>
            </>
          ) : (
            <>
              <svg
                className="h-6 w-6"
                viewBox="0 -0.5 25 25"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                <g
                  id="SVGRepo_tracerCarrier"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                ></g>
                <g id="SVGRepo_iconCarrier">
                  {" "}
                  <path
                    d="M12.25 18.25C15.0114 18.25 17.25 16.0114 17.25 13.25C17.25 10.4886 15.0114 8.25 12.25 8.25C9.48858 8.25 7.25 10.4886 7.25 13.25C7.25 16.0114 9.48858 18.25 12.25 18.25Z"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                  <path
                    d="M2.25001 10C2.24847 9.13152 2.52963 8.28612 3.051 7.59155C3.57238 6.89699 4.30564 6.39103 5.14001 6.15002C5.96585 5.90841 6.69178 5.40688 7.21001 4.71997L8.05001 3.59998C8.4226 3.10319 8.90574 2.69995 9.46116 2.42224C10.0166 2.14453 10.629 2 11.25 2H13.25C13.871 2 14.4834 2.14453 15.0389 2.42224C15.5943 2.69995 16.0774 3.10319 16.45 3.59998L17.3 4.71997C17.8141 5.41113 18.5416 5.91375 19.37 6.15002C20.2025 6.39284 20.9336 6.89959 21.453 7.59399C21.9725 8.2884 22.2522 9.13281 22.25 10L21.58 18C21.4726 19.081 20.9726 20.0851 20.1746 20.8223C19.3766 21.5594 18.3361 21.9784 17.25 22H7.25001C6.16389 21.9784 5.12339 21.5594 4.32543 20.8223C3.52747 20.0851 3.02746 19.081 2.92002 18L2.25001 10Z"
                    stroke="#ffffff"
                    stroke-width="1.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></path>{" "}
                </g>
              </svg>
              <span>Start Camera</span>
            </>
          )}
        </button>

        <div className="text-xs text-gray-400">
          Emotion detection {isCameraOn ? "active" : "paused"}
        </div>
      </div>
    </div>
  );
};

export default Detection;
