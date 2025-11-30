import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Info, Activity } from 'lucide-react';

const Prediction = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        age: 50,
        gender: 'Male',
        glucose: 100,
        bmi: 25,
        work: 'Private',
        smoking: 'never smoked',
        hypertension: false,
        heart_disease: false,
        residence: 'Urban',
        marital: 'Yes'
    });

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const calculateBMI = () => {
        const bmi = parseFloat(formData.bmi);
        if (bmi < 18.5) return 'Underweight';
        if (bmi < 25) return 'Normal weight';
        if (bmi < 30) return 'Overweight';
        return 'Obese';
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            // Prepare data for API
            const apiData = {
                gender: formData.gender,
                age: parseFloat(formData.age),
                hypertension: formData.hypertension ? 1 : 0,
                heart_disease: formData.heart_disease ? 1 : 0,
                ever_married: formData.marital,
                work_type: formData.work,
                Residence_type: formData.residence,
                avg_glucose_level: parseFloat(formData.glucose),
                bmi: parseFloat(formData.bmi) || 0,
                smoking_status: formData.smoking
            };

            // Call the ML model API
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(apiData)
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const result = await response.json();

            // Navigate to results with API response
            navigate('/results', {
                state: {
                    probability: result.probability,
                    isHighRisk: result.is_high_risk,
                    riskFactors: result.risk_factors,
                    data: formData
                }
            });
        } catch (error) {
            console.error('Error calling API:', error);
            alert('Failed to get prediction. Please make sure the API is running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container" style={{ padding: '0 1rem' }}>
            <div className="glass" style={{ borderRadius: '1.5rem', padding: '2rem', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)' }}>
                <div style={{ marginBottom: '2rem' }}>
                    <h2 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>Risk Assessment</h2>
                    <p style={{ color: 'var(--text-muted)' }}>Complete the form below to get your stroke risk analysis.</p>
                </div>

                <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    {/* Personal Information Section */}
                    <div>
                        <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'white', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ width: '2rem', height: '2rem', borderRadius: '50%', background: 'var(--primary-500)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.875rem' }}>1</span>
                            Personal Information
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                <label className="flex items-center gap-2" style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1' }}>
                                    Age <Info style={{ width: '1rem', height: '1rem', color: '#64748b' }} />
                                </label>
                                <input
                                    type="number"
                                    name="age"
                                    value={formData.age}
                                    onChange={handleChange}
                                    className="glass-input"
                                    style={{ width: '100%', borderRadius: '0.75rem', padding: '0.75rem 1rem' }}
                                    min="1" max="120"
                                    required
                                />
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1' }}>Gender</label>
                                <select name="gender" value={formData.gender} onChange={handleChange} className="glass-input" style={{ width: '100%', borderRadius: '0.75rem', padding: '0.75rem 1rem' }}>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Health Metrics Section */}
                    <div>
                        <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'white', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ width: '2rem', height: '2rem', borderRadius: '50%', background: 'var(--primary-500)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.875rem' }}>2</span>
                            Health Metrics
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1' }}>Avg Glucose Level (mg/dL)</label>
                                <input
                                    type="number"
                                    name="glucose"
                                    value={formData.glucose}
                                    onChange={handleChange}
                                    className="glass-input"
                                    style={{ width: '100%', borderRadius: '0.75rem', padding: '0.75rem 1rem' }}
                                    required
                                />
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1' }}>BMI ({calculateBMI()})</label>
                                <input
                                    type="number"
                                    name="bmi"
                                    value={formData.bmi}
                                    onChange={handleChange}
                                    className="glass-input"
                                    style={{ width: '100%', borderRadius: '0.75rem', padding: '0.75rem 1rem' }}
                                    step="0.1"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Lifestyle & Medical History Section */}
                    <div>
                        <h3 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'white', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ width: '2rem', height: '2rem', borderRadius: '50%', background: 'var(--primary-500)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.875rem' }}>3</span>
                            Lifestyle & Medical History
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6" style={{ marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1' }}>Work Type</label>
                                <select name="work" value={formData.work} onChange={handleChange} className="glass-input" style={{ width: '100%', borderRadius: '0.75rem', padding: '0.75rem 1rem' }}>
                                    <option value="Private">Private Sector</option>
                                    <option value="Self-employed">Self Employed</option>
                                    <option value="Govt_job">Government</option>
                                    <option value="children">Student/Child</option>
                                </select>
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1' }}>Smoking Status</label>
                                <select name="smoking" value={formData.smoking} onChange={handleChange} className="glass-input" style={{ width: '100%', borderRadius: '0.75rem', padding: '0.75rem 1rem' }}>
                                    <option value="never smoked">Never Smoked</option>
                                    <option value="formerly smoked">Formerly Smoked</option>
                                    <option value="smokes">Currently Smokes</option>
                                    <option value="Unknown">Unknown</option>
                                </select>
                            </div>
                        </div>

                        <div style={{ padding: '1rem', borderRadius: '0.75rem', backgroundColor: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(71, 85, 105, 0.5)', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            <p style={{ fontSize: '0.875rem', fontWeight: 500, color: '#cbd5e1', marginBottom: '0.5rem' }}>Medical History</p>
                            <label className="flex items-center gap-3" style={{ cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    name="hypertension"
                                    checked={formData.hypertension}
                                    onChange={handleChange}
                                    style={{ width: '1.25rem', height: '1.25rem', borderRadius: '0.25rem', border: '1px solid #475569', backgroundColor: '#334155', accentColor: 'var(--primary-500)' }}
                                />
                                <span style={{ color: '#cbd5e1' }}>Hypertension (High Blood Pressure)</span>
                            </label>
                            <label className="flex items-center gap-3" style={{ cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    name="heart_disease"
                                    checked={formData.heart_disease}
                                    onChange={handleChange}
                                    style={{ width: '1.25rem', height: '1.25rem', borderRadius: '0.25rem', border: '1px solid #475569', backgroundColor: '#334155', accentColor: 'var(--primary-500)' }}
                                />
                                <span style={{ color: '#cbd5e1' }}>Heart Disease</span>
                            </label>
                        </div>
                    </div>

                    <div style={{ paddingTop: '1.5rem', borderTop: '1px solid rgba(255, 255, 255, 0.1)', display: 'flex', justifyContent: 'center' }}>
                        <button
                            type="submit"
                            disabled={loading}
                            className="btn btn-primary"
                            style={{ opacity: loading ? 0.7 : 1, fontSize: '1.125rem', padding: '1rem 2rem' }}
                        >
                            {loading ? 'Analyzing...' : 'Analyze Risk'}
                            {!loading && <Activity style={{ width: '1.25rem', height: '1.25rem' }} />}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default Prediction;
