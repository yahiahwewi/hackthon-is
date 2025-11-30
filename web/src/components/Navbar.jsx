import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Brain, Activity, Home } from 'lucide-react';

const Navbar = () => {
    const location = useLocation();

    const isActive = (path) => location.pathname === path;

    return (
        <nav className="container" style={{ marginBottom: '2rem', paddingTop: '1.5rem' }}>
            <div className="glass" style={{ borderRadius: '1rem', padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Link to="/" className="flex items-center gap-3" style={{ textDecoration: 'none' }}>
                    <div className="bg-gradient-circle" style={{ width: '2.5rem', height: '2.5rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Brain className="text-white" style={{ width: '1.5rem', height: '1.5rem', color: 'white' }} />
                    </div>
                    <div>
                        <h1 className="text-gradient" style={{ fontSize: '1.25rem', fontWeight: 'bold', margin: 0 }}>
                            NeuroGuard AI
                        </h1>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }} className="sm:block hidden">Stroke Risk Assessment</p>
                    </div>
                </Link>

                <div className="flex gap-2">
                    <Link
                        to="/"
                        className="flex items-center gap-2"
                        style={{
                            padding: '0.75rem',
                            borderRadius: '0.75rem',
                            textDecoration: 'none',
                            backgroundColor: isActive('/') ? 'rgba(14, 165, 233, 0.2)' : 'transparent',
                            color: isActive('/') ? 'var(--primary-400)' : 'var(--text-muted)',
                            transition: 'all 0.3s ease'
                        }}
                    >
                        <Home style={{ width: '1.25rem', height: '1.25rem' }} />
                        <span className="sm:inline hidden">Home</span>
                    </Link>
                    <Link
                        to="/predict"
                        className="flex items-center gap-2"
                        style={{
                            padding: '0.75rem',
                            borderRadius: '0.75rem',
                            textDecoration: 'none',
                            backgroundColor: isActive('/predict') ? 'rgba(14, 165, 233, 0.2)' : 'transparent',
                            color: isActive('/predict') ? 'var(--primary-400)' : 'var(--text-muted)',
                            transition: 'all 0.3s ease'
                        }}
                    >
                        <Activity style={{ width: '1.25rem', height: '1.25rem' }} />
                        <span className="sm:inline hidden">Assess Risk</span>
                    </Link>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
