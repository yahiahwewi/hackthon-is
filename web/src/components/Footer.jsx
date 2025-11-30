import React from 'react';
import { Trophy, Heart } from 'lucide-react';

const Footer = () => {
    return (
        <footer className="container animate-fade-in" style={{ marginTop: '3rem', marginBottom: '2rem', animationDelay: '0.3s' }}>
            <div className="glass" style={{ borderRadius: '1rem', padding: '1.5rem', textAlign: 'center' }}>
                <div className="flex flex-col md:flex-row items-center justify-center gap-6">
                    <div className="flex items-center gap-4">
                        <div className="bg-gradient-circle animate-pulse-slow" style={{ width: '3rem', height: '3rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 10px 15px -3px rgba(14, 165, 233, 0.2)' }}>
                            <Trophy style={{ color: 'white', width: '1.5rem', height: '1.5rem' }} />
                        </div>
                        <div style={{ textAlign: 'left' }}>
                            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.25rem', margin: 0 }}>
                                Made with <Heart style={{ color: '#f87171', width: '0.75rem', height: '0.75rem', fill: 'currentColor' }} /> by
                            </p>
                            <h3 className="text-gradient" style={{ fontSize: '1.25rem', fontWeight: 'bold', margin: 0 }}>
                                Team EA over the ROC
                            </h3>
                        </div>
                    </div>

                    <div className="hidden md:block" style={{ width: '1px', height: '3rem', background: 'linear-gradient(to bottom, transparent, var(--slate-850), transparent)' }}></div>

                    <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                        <p style={{ fontWeight: 500, color: '#cbd5e1', margin: 0 }}>Achieving Excellence in Healthcare AI</p>
                        <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem', fontFamily: 'monospace', margin: 0 }}>ROC-AUC: 0.84+ | Precision Medicine</p>
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
