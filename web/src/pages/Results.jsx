import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, RefreshCcw, Activity } from 'lucide-react';

const Results = () => {
    const location = useLocation();
    const { probability, isHighRisk, riskFactors, data } = location.state || {
        probability: 0,
        isHighRisk: false,
        riskFactors: [],
        data: {}
    };

    const percentage = Math.round(probability * 100);

    return (
        <div className="container" style={{ padding: '0 1rem' }}>
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass text-center"
                style={{ borderRadius: '1.5rem', padding: '2rem', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)' }}
            >
                <div style={{ marginBottom: '2rem' }}>
                    <div style={{
                        width: '6rem',
                        height: '6rem',
                        margin: '0 auto',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '1.5rem',
                        backgroundColor: isHighRisk ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                        color: isHighRisk ? '#f87171' : '#4ade80'
                    }}>
                        <Activity style={{ width: '3rem', height: '3rem' }} />
                    </div>

                    <h2 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>Analysis Complete</h2>
                    <p style={{ color: 'var(--text-muted)' }}>Based on the provided health metrics</p>
                </div>

                <div style={{ marginBottom: '3rem', position: 'relative' }}>
                    <div className="text-gradient" style={{ fontSize: '3.75rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                        {percentage}%
                    </div>
                    <p style={{
                        fontSize: '1.25rem',
                        fontWeight: 500,
                        color: isHighRisk ? '#f87171' : '#4ade80'
                    }}>
                        {isHighRisk ? 'High Risk Detected' : 'Low Risk Profile'}
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6" style={{ marginBottom: '3rem', textAlign: 'left' }}>
                    <div style={{ padding: '1.5rem', borderRadius: '1rem', backgroundColor: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(71, 85, 105, 0.5)' }}>
                        <h3 className="flex items-center gap-2" style={{ fontSize: '1.125rem', fontWeight: 600, color: 'white', marginBottom: '1rem' }}>
                            <AlertTriangle style={{ width: '1.25rem', height: '1.25rem', color: '#eab308' }} /> Risk Factors
                        </h3>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem', color: '#cbd5e1' }}>
                            {riskFactors && riskFactors.length > 0 ? (
                                riskFactors.map((factor, index) => (
                                    <li key={index} className="flex items-center gap-2">
                                        <div style={{ width: '0.375rem', height: '0.375rem', borderRadius: '50%', backgroundColor: '#f87171' }}></div>
                                        {factor}
                                    </li>
                                ))
                            ) : (
                                <li>No significant risk factors identified.</li>
                            )}
                        </ul>
                    </div>

                    <div style={{ padding: '1.5rem', borderRadius: '1rem', backgroundColor: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(71, 85, 105, 0.5)' }}>
                        <h3 className="flex items-center gap-2" style={{ fontSize: '1.125rem', fontWeight: 600, color: 'white', marginBottom: '1rem' }}>
                            <CheckCircle style={{ width: '1.25rem', height: '1.25rem', color: '#22c55e' }} /> Recommendations
                        </h3>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem', color: '#cbd5e1' }}>
                            <li className="flex items-center gap-2">
                                <div style={{ width: '0.375rem', height: '0.375rem', borderRadius: '50%', backgroundColor: '#4ade80' }}></div>
                                Maintain regular exercise
                            </li>
                            <li className="flex items-center gap-2">
                                <div style={{ width: '0.375rem', height: '0.375rem', borderRadius: '50%', backgroundColor: '#4ade80' }}></div>
                                Monitor blood pressure regularly
                            </li>
                            <li className="flex items-center gap-2">
                                <div style={{ width: '0.375rem', height: '0.375rem', borderRadius: '50%', backgroundColor: '#4ade80' }}></div>
                                Maintain a healthy diet
                            </li>
                            <li className="flex items-center gap-2">
                                <div style={{ width: '0.375rem', height: '0.375rem', borderRadius: '50%', backgroundColor: '#4ade80' }}></div>
                                Schedule yearly health checkups
                            </li>
                            {isHighRisk && (
                                <li className="flex items-center gap-2">
                                    <div style={{ width: '0.375rem', height: '0.375rem', borderRadius: '50%', backgroundColor: '#4ade80' }}></div>
                                    Consult with a healthcare professional
                                </li>
                            )}
                        </ul>
                    </div>
                </div>

                <Link
                    to="/predict"
                    className="btn btn-secondary"
                    style={{ textDecoration: 'none', display: 'inline-flex' }}
                >
                    <RefreshCcw style={{ width: '1rem', height: '1rem' }} /> Start New Assessment
                </Link>
            </motion.div>
        </div>
    );
};

export default Results;
