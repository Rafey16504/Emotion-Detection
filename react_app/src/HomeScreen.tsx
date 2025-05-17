import React from 'react';
import { Link } from 'react-router-dom';

const HomeScreen: React.FC = () => {
  return (
    <div className="flex flex-col justify-center items-center h-screen bg-gradient-to-r from-blue-500 to-purple-500 text-white">
      <h1 className="text-3xl underline mb-6 font-semibold">Welcome to Emotion Detection</h1>
      <p className="text-lg mb-8">Detect emotions in real-time using your webcam.</p>
      <Link
        to="/camera"
        className="bg-white text-blue-500 px-6 py-3 rounded-full font-semibold hover:bg-gray-100 transition duration-300"
      >
        Start Detection
      </Link>
    </div>
  );
};

export default HomeScreen;