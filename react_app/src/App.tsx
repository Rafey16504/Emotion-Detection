import { useEffect, useRef } from 'react';
import './App.css';

function App() {
  const videoRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.src = 'http://localhost:5000/video_feed';
    }
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Real-time Emotion Detection</h1>
      <img 
        ref={videoRef} 
        className="rounded-lg shadow-lg"
        alt="Live emotion detection feed"
      />
    </div>
  );
}

export default App;