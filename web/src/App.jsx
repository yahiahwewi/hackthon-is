import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Prediction from './pages/Prediction';
import Results from './pages/Results';

function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
        {/* Background Gradients */}
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', overflow: 'hidden', zIndex: -10, pointerEvents: 'none' }}>
          <div className="bg-blob" style={{ top: '-10%', left: '-10%', width: '24rem', height: '24rem', background: 'var(--primary-500)' }}></div>
          <div className="bg-blob" style={{ bottom: '-10%', right: '-10%', width: '24rem', height: '24rem', background: 'var(--accent)' }}></div>
          <div className="bg-blob" style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '24rem', height: '24rem', background: 'var(--medical-500)', opacity: 0.05 }}></div>
        </div>

        <Navbar />

        <main style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Prediction />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </main>

        <Footer />
      </div>
    </Router>
  );
}

export default App;
