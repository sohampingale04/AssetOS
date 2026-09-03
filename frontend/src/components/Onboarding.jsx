import { useState } from 'react';
import { ArrowRight, Loader2 } from 'lucide-react';

export default function Onboarding({ setProfile, setPortfolio }) {
    const [loading, setLoading] = useState(false);
    const [step, setStep] = useState(1);
    const [ipsData, setIpsData] = useState(null);
    const [savedUserData, setSavedUserData] = useState(null);

    const [formData, setFormData] = useState({
        name: '',
        age: '',
        income: '',
        net_worth: '',
        capital: '',
        goal: 'Wealth Preservation',
        goal_years: '',
        risk_attitude: '3',
        loss_tolerance: '',
        liquidity: ''
    });

    const handleGenerateIPS = async (e) => {
        e.preventDefault();
        
        const cleanData = { ...formData };
        ['capital', 'income', 'net_worth'].forEach(field => {
            if (cleanData[field]) {
                cleanData[field] = cleanData[field].toString().replace(/,/g, '');
            }
        });

        // Validation
        const cap = parseFloat(cleanData.capital);
        if (isNaN(cap) || cap <= 0) {
            alert("Investment Capital must be greater than zero.");
            return;
        }

        setLoading(true);

        try {
            // 1. Create Profile and Receive IPS constraints
            const userRes = await fetch('/api/onboard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cleanData)
            });
            const userData = await userRes.json();

            if (!userRes.ok) throw new Error(userData.error || "Onboarding failed");

            // Save user data and proceed to IPS confirmation
            setSavedUserData(userData.profile);
            setIpsData(userData.ips);
            setStep(2);

        } catch (err) {
            console.error("Error during initialization:", err);
            alert("Error: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGeneratePortfolio = async () => {
        setLoading(true);
        try {
            // 2. Generate Portfolio
            const portRes = await fetch('/api/generate_portfolio', { method: 'POST' });
            const portText = await portRes.text();

            let portData;
            try {
                portData = JSON.parse(portText);
            } catch (jsonErr) {
                throw new Error("Invalid JSON response from server: " + portText.substring(0, 100));
            }

            if (!portRes.ok) throw new Error(portData.error || "Portfolio generation failed");

            // 3. Set State (Success)
            setProfile(savedUserData);
            setPortfolio(portData);

        } catch (err) {
            console.error("Error generating portfolio:", err);
            alert("Error: " + err.message);
            setLoading(false); // Only stop loading if error; otherwise App routes away.
        }
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[70vh] text-elite-gold">
                <Loader2 className="w-12 h-12 animate-spin mb-4" />
                <h2 className="text-2xl font-serif">Structuring Assets...</h2>
                <p className="text-elite-slate mt-2">{step === 1 ? 'Generating Institutional Policy' : 'Running Monte Carlo Simulations'}</p>
            </div>
        );
    }

    if (step === 2 && ipsData) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[70vh] py-10">
                <div className="w-full max-w-2xl">
                    <div className="text-center mb-10">
                        <h1 className="text-4xl md:text-5xl mb-4 bg-clip-text text-transparent bg-gradient-to-r from-elite-cream to-elite-slate">
                            Investment Policy Statement
                        </h1>
                        <p className="text-elite-gold/80 italic font-serif text-xl">
                            Please review the dynamically generated constraints.
                        </p>
                    </div>
                    
                    <div className="glass-card p-10 space-y-6 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-elite-gold to-transparent opacity-50" />
                        
                        <div className="space-y-4">
                            <div className="flex justify-between border-b border-elite-gold/20 pb-4">
                                <span className="text-elite-slate uppercase tracking-widest text-sm">Target Return</span>
                                <span className="text-elite-gold font-mono text-lg font-bold">{(ipsData.ReturnObjective * 100).toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between border-b border-elite-gold/20 pb-4">
                                <span className="text-elite-slate uppercase tracking-widest text-sm">Equity Bounds</span>
                                <span className="text-elite-gold font-mono text-lg font-bold">{(ipsData.EquityMin * 100).toFixed(0)}% - {(ipsData.EquityMax * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between border-b border-elite-gold/20 pb-4">
                                <span className="text-elite-slate uppercase tracking-widest text-sm">Cash Requirement</span>
                                <span className="text-elite-gold font-mono text-lg font-bold">{(ipsData.CashMin * 100).toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between border-b border-elite-gold/20 pb-4">
                                <span className="text-elite-slate uppercase tracking-widest text-sm">Max Portfolio Drawdown Guard</span>
                                <span className="text-elite-gold font-mono text-lg font-bold">{(ipsData.MaxDrawdown * 100).toFixed(1)}%</span>
                            </div>
                        </div>

                        <div className="pt-10 flex justify-between items-center">
                            <button 
                                onClick={() => setStep(1)}
                                className="px-6 py-2 text-elite-gold border border-elite-gold/30 hover:bg-elite-gold/10 transition-colors uppercase tracking-widest text-xs"
                            >
                                ← Edit Inputs
                            </button>
                            <button 
                                onClick={handleGeneratePortfolio}
                                className="gold-btn-filled flex items-center space-x-2 group"
                            >
                                <span>Initialize Portfolio</span>
                                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col items-center justify-center min-h-[70vh] py-10">
            <div className="w-full max-w-4xl">
                <div className="text-center mb-10">
                    <h1 className="text-5xl md:text-6xl mb-4 bg-clip-text text-transparent bg-gradient-to-r from-elite-cream to-elite-slate">
                        Welcome, Investor.
                    </h1>
                    <p className="text-elite-gold/80 italic font-serif text-xl">
                        "Fortune favors the bold, but wealth favors the disciplined."
                    </p>
                </div>

                <form onSubmit={handleGenerateIPS} className="glass-card p-10 space-y-6 relative overflow-hidden">
                    {/* Decorative Gold Line */}
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-elite-gold to-transparent opacity-50" />

                    {/* Row 1 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Client Name</label>
                            <input type="text" required placeholder="e.g. Alexandra" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} className="input-field" />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Age</label>
                            <input type="number" required placeholder="e.g. 45" value={formData.age} onChange={e => setFormData({ ...formData, age: e.target.value })} className="input-field" />
                        </div>
                    </div>

                    {/* Row 2 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Annual Income (₹)</label>
                            <input type="text" required placeholder="e.g. 15,000,000" value={formData.income} onChange={e => setFormData({ ...formData, income: e.target.value })} className="input-field" />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Net Worth (₹)</label>
                            <input type="text" required placeholder="e.g. 50,000,000" value={formData.net_worth} onChange={e => setFormData({ ...formData, net_worth: e.target.value })} className="input-field" />
                        </div>
                    </div>

                    {/* Row 3 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Investment Capital (₹)</label>
                            <input type="text" required placeholder="e.g. 10,000,000" value={formData.capital} onChange={e => setFormData({ ...formData, capital: e.target.value })} className="input-field" />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Primary Goal</label>
                            <select value={formData.goal} onChange={e => setFormData({ ...formData, goal: e.target.value })} className="input-field appearance-none">
                                <option>Wealth Preservation</option>
                                <option>Aggressive Growth</option>
                                <option>Balanced Income</option>
                            </select>
                        </div>
                    </div>

                    {/* Row 4 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Years to Goal</label>
                            <input type="number" required placeholder="e.g. 10" value={formData.goal_years} onChange={e => setFormData({ ...formData, goal_years: e.target.value })} className="input-field" />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Risk Attitude (1-5)</label>
                            <select value={formData.risk_attitude} onChange={e => setFormData({ ...formData, risk_attitude: e.target.value })} className="input-field appearance-none p-2 w-full">
                                <option value="1">1 - Very Conservative</option>
                                <option value="2">2 - Conservative</option>
                                <option value="3">3 - Moderate</option>
                                <option value="4">4 - Aggressive</option>
                                <option value="5">5 - Very Aggressive</option>
                            </select>
                        </div>
                    </div>

                    {/* Row 5 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Max Loss Tolerance (%)</label>
                            <input type="number" required placeholder="e.g. 15" min="0" max="100" value={formData.loss_tolerance} onChange={e => setFormData({ ...formData, loss_tolerance: e.target.value })} className="input-field" />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs uppercase tracking-widest text-elite-gold">Liquidity Need (%)</label>
                            <input type="number" required placeholder="e.g. 5" min="0" max="100" value={formData.liquidity} onChange={e => setFormData({ ...formData, liquidity: e.target.value })} className="input-field" />
                        </div>
                    </div>

                    <div className="pt-4 flex justify-center">
                        <button type="submit" disabled={loading} className="gold-btn-filled flex items-center space-x-2 group">
                            {loading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Compiling Logic...</span>
                                </>
                            ) : (
                                <>
                                    <span>Review IPS Policy</span>
                                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                                </>
                            )}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
