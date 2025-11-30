import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, ShieldCheck, Activity, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

const Home = () => {
    return (
        <div className="container flex flex-col items-center justify-center" style={{ minHeight: '60vh', padding: '0 1rem' }}>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center"
                style={{ maxWidth: '48rem' }}
            >
                <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.5rem 1rem',
                    borderRadius: '9999px',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    border: '1px solid rgba(14, 165, 233, 0.2)',
                    color: 'var(--primary-400)',
                    marginBottom: '1.5rem'
                }}>
                    <span style={{ position: 'relative', display: 'flex', height: '0.5rem', width: '0.5rem' }}>
                        <span style={{ position: 'absolute', display: 'inline-flex', height: '100%', width: '100%', borderRadius: '50%', backgroundColor: 'var(--primary-400)', opacity: 0.75, animation: 'ping 1s cubic-bezier(0, 0, 0.2, 1) infinite' }}></span>
                        <span style={{ position: 'relative', display: 'inline-flex', borderRadius: '50%', height: '0.5rem', width: '0.5rem', backgroundColor: 'var(--primary-500)' }}></span>
                    </span>
                    AI-Powered Health Analysis
                </div>

                <h1 style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '1.5rem', lineHeight: 1.2 }}>
                    Protect Your <br />
                    <span className="text-gradient">
                        Brain Health
                    </span>
                </h1>

                <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', marginBottom: '2.5rem', lineHeight: 1.6 }}>
                    Advanced stroke prediction using state-of-the-art machine learning algorithms.
                    Get a personalized risk assessment in minutes.
                </p>

                <div className="flex flex-col sm:flex-row justify-center gap-4" style={{ marginBottom: '4rem' }}>
                    <Link to="/predict" className="btn btn-primary">
                        Start Assessment <ArrowRight style={{ width: '1.25rem', height: '1.25rem' }} />
                    </Link>
                    <button className="btn btn-secondary">
                        Learn More
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center md:text-left">
                    <div className="glass" style={{ padding: '1.5rem', borderRadius: '1rem' }}>
                        <div style={{ width: '3rem', height: '3rem', borderRadius: '0.75rem', backgroundColor: 'rgba(14, 165, 233, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: 'var(--primary-400)' }}>
                            <Brain style={{ width: '1.5rem', height: '1.5rem' }} />
                        </div>
                        <h3 style={{ fontSize: '1.125rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>Advanced AI</h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Powered by ensemble machine learning models with high accuracy.</p>
                    </div>
                    <div className="glass" style={{ padding: '1.5rem', borderRadius: '1rem' }}>
                        <div style={{ width: '3rem', height: '3rem', borderRadius: '0.75rem', backgroundColor: 'rgba(34, 197, 94, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: 'var(--medical-400)' }}>
                            <ShieldCheck style={{ width: '1.5rem', height: '1.5rem' }} />
                        </div>
                        <h3 style={{ fontSize: '1.125rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>Early Detection</h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Identify potential risks early to take preventive measures.</p>
                    </div>
                    <div className="glass" style={{ padding: '1.5rem', borderRadius: '1rem' }}>
                        <div style={{ width: '3rem', height: '3rem', borderRadius: '0.75rem', backgroundColor: 'rgba(129, 140, 248, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: 'var(--accent)' }}>
                            <Activity style={{ width: '1.5rem', height: '1.5rem' }} />
                        </div>
                        <h3 style={{ fontSize: '1.125rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>Real-time Analysis</h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Instant processing of your health metrics and lifestyle data.</p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default Home;
