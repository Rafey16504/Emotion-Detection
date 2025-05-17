import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Home from './Home';
import Detection from './Detection';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/detection" element={<Detection />} />
      </Routes>
    </Router>
  );
}

export default App;